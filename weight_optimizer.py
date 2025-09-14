import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging
from typing import Dict, Tuple, List, Optional
from config import DEFAULT_FEATURE_WEIGHTS

# ============================================================================
# AGGRESSIVENESS CONTROL SETTING  
# ============================================================================
# Single parameter to control how aggressively the optimizer fits to your ratings:
# 
# 0.1-0.3 = CONSERVATIVE (less risk of overfitting, may underfit)
# 0.4-0.6 = MODERATE (balanced approach)  
# 0.7-0.9 = AGGRESSIVE (fits closely to ratings, higher overfitting risk)

AGGRESSIVENESS = 0.4  # Single control knob - reasonable moderate default

# Other settings
WEIGHT_BOUNDS = (-50, 50)
MIN_RATINGS = 10

# Internal parameters (automatically calculated from AGGRESSIVENESS)
def _get_optimization_params(aggressiveness: float):
    """Convert single aggressiveness parameter to internal optimization parameters"""
    # Make the mapping safer to avoid overflow - cap the aggressiveness effect
    safe_aggr = min(0.9, max(0.1, aggressiveness))  # Clamp to safe range
    
    return {
        # L2 regularization: More conservative mapping to prevent overflow
        'l2_regularization': max(0.01, 0.1 * (1.0 / safe_aggr)),  # Range: 0.01 to 1.0
        
        # Learning rate: Moderate range
        'learning_rate': min(0.9, 0.5 + 0.4 * safe_aggr),  # Range: 0.5 to 0.9
        
        # Ranking margin: Conservative range  
        'ranking_margin': 0.8 + 0.8 * safe_aggr  # Range: 0.8 to 1.6
    }

logger = logging.getLogger(__name__)


class WeightOptimizer:
    """
    Optimizes feature weights using pairwise ranking loss to improve property scoring alignment with user ratings.
    """
    
    def __init__(self, 
                 aggressiveness: float = AGGRESSIVENESS,
                 weight_bounds: Tuple[float, float] = WEIGHT_BOUNDS,
                 min_ratings: int = MIN_RATINGS):
        """
        Initialize the WeightOptimizer.
        
        Args:
            aggressiveness: Single parameter controlling optimization aggressiveness (0.1=conservative, 0.9=aggressive)
            weight_bounds: Tuple of (min, max) bounds for all weights
            min_ratings: Minimum number of ratings required to run optimization
        """
        # Get internal parameters from aggressiveness setting (safer mapping this time)
        params = _get_optimization_params(aggressiveness)
        
        self.aggressiveness = aggressiveness
        self.weight_bounds = weight_bounds
        self.l2_regularization = params['l2_regularization']
        self.learning_rate = params['learning_rate'] 
        self.ranking_margin = params['ranking_margin']
        self.min_ratings = min_ratings
        self.rating_values = {'yes': 2, 'maybe': 1, 'no': 0}
        
        # Log the effective parameter values
        logger.info(f"WeightOptimizer initialized with aggressiveness={aggressiveness:.2f}")
        logger.info(f"  -> L2 regularization: {self.l2_regularization:.4f} (prevents overflow)")
        logger.info(f"  -> Learning rate: {self.learning_rate:.2f}")  
        logger.info(f"  -> Ranking margin: {self.ranking_margin:.2f}")
        
    def extract_feature_scores(self, properties_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract feature scores from properties DataFrame.
        
        Args:
            properties_df: DataFrame with feature score columns (ending in '_score')
            
        Returns:
            Tuple of (feature_scores_matrix, feature_names)
        """
        # Find all columns ending with '_score'
        score_columns = [col for col in properties_df.columns if col.endswith('_score')]
        
        # Filter to only features that exist in DEFAULT_FEATURE_WEIGHTS
        valid_features = []
        for col in score_columns:
            feature_name = col.replace('_score', '')
            if feature_name in DEFAULT_FEATURE_WEIGHTS:
                valid_features.append(col)
        
        if not valid_features:
            raise ValueError("No valid feature score columns found in properties DataFrame")
            
        # Extract feature matrix
        feature_matrix = properties_df[valid_features].fillna(50.0).values  # Fill NaN with neutral score
        feature_names = [col.replace('_score', '') for col in valid_features]
        
        logger.info(f"Extracted {len(feature_names)} features: {feature_names}")
        return feature_matrix, feature_names
        
    def create_pairwise_comparisons(self, properties_df: pd.DataFrame, ratings_dict: Dict[int, str]) -> List[Tuple[int, int, float]]:
        """
        Create pairwise comparisons from ratings where ratings differ.
        
        Args:
            properties_df: DataFrame with zpid column
            ratings_dict: Dictionary mapping zpid to rating ('yes', 'maybe', 'no')
            
        Returns:
            List of (idx_i, idx_j, preference_diff) tuples where preference_diff > 0 means i should rank higher than j
        """
        # Create zpid to index mapping
        zpid_to_idx = {zpid: idx for idx, zpid in enumerate(properties_df['zpid'])}
        
        # Filter ratings to only include rated properties that exist in our dataframe
        valid_ratings = {}
        for zpid, rating in ratings_dict.items():
            if zpid in zpid_to_idx and rating in self.rating_values:
                valid_ratings[zpid] = rating
        
        if len(valid_ratings) < self.min_ratings:
            raise ValueError(f"Insufficient ratings: {len(valid_ratings)} < {self.min_ratings}")
        
        # Create pairwise comparisons
        comparisons = []
        rated_zpids = list(valid_ratings.keys())
        
        for i, zpid_i in enumerate(rated_zpids):
            for j, zpid_j in enumerate(rated_zpids[i+1:], i+1):
                rating_i = self.rating_values[valid_ratings[zpid_i]]
                rating_j = self.rating_values[valid_ratings[zpid_j]]
                
                if rating_i != rating_j:
                    idx_i = zpid_to_idx[zpid_i]
                    idx_j = zpid_to_idx[zpid_j]
                    preference_diff = rating_i - rating_j  # Positive means i is preferred over j
                    comparisons.append((idx_i, idx_j, preference_diff))
        
        logger.info(f"Created {len(comparisons)} pairwise comparisons from {len(valid_ratings)} ratings")
        return comparisons
        
    def pairwise_ranking_loss(self, weights: np.ndarray, 
                             feature_matrix: np.ndarray, 
                             comparisons: List[Tuple[int, int, float]],
                             current_weights: np.ndarray,
                             feature_names: List[str]) -> float:
        """
        Calculate pairwise ranking loss with L2 regularization.
        
        Args:
            weights: Current weight values being optimized
            feature_matrix: Matrix of feature scores (properties x features)
            comparisons: List of (idx_i, idx_j, preference_diff) tuples
            current_weights: Current/default weights for regularization
            feature_names: List of feature names to determine sign handling
            
        Returns:
            Loss value to minimize
        """
        # Calculate property scores using current weights
        scores = self.calculate_property_scores(weights, feature_matrix, feature_names)
        
        # Pairwise ranking loss
        ranking_loss = 0.0
        for idx_i, idx_j, preference_diff in comparisons:
            score_diff = scores[idx_i] - scores[idx_j]  # Positive means i scores higher than j
            
            # We want score_diff to have the same sign as preference_diff
            # If preference_diff > 0 (i preferred), we want score_diff > 0
            # Use hinge loss: max(0, -preference_diff * score_diff + margin)
            loss = max(0, -preference_diff * score_diff + self.ranking_margin)
            ranking_loss += loss
        
        # L2 regularization scaled by number of samples
        n_comparisons = len(comparisons)
        regularization = self.l2_regularization * n_comparisons * np.sum((weights - current_weights) ** 2)
        
        total_loss = ranking_loss + regularization
        return total_loss
        
    def calculate_property_scores(self, weights: np.ndarray, feature_matrix: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Calculate total property scores given weights and features.
        
        Args:
            weights: Weight values
            feature_matrix: Matrix of feature scores (0-100 scale)
            feature_names: List of feature names to determine sign handling
            
        Returns:
            Array of total scores for each property
        """
        scores = np.zeros(len(feature_matrix))
        
        for i, (weight, feature_name) in enumerate(zip(weights, feature_names)):
            feature_scores = feature_matrix[:, i]
            
            # Handle negative features (where higher values are worse)
            # Negative features in DEFAULT_FEATURE_WEIGHTS: road_exposure, avg_risk_severity, avg_crime_severity, avg_school_distance, drive_time
            # For these, we want to invert the scoring when the weight is positive
            negative_features = {'road_exposure', 'avg_risk_severity', 'avg_crime_severity', 'avg_school_distance', 'drive_time'}
            
            if feature_name in negative_features:
                # For negative features, flip the score when weight is positive
                if weight > 0:
                    contribution = (100 - feature_scores) * weight
                else:
                    contribution = feature_scores * abs(weight)
            else:
                # For positive features, use scores directly
                if weight > 0:
                    contribution = feature_scores * weight
                else:
                    contribution = (100 - feature_scores) * abs(weight)
            
            scores += contribution
        
        return scores
        
    def optimize_weights(self, properties_df: pd.DataFrame, ratings_dict: Dict[int, str]) -> Tuple[Dict[str, float], Dict[str, any]]:
        """
        Optimize feature weights based on user ratings using pairwise ranking loss.
        
        Args:
            properties_df: DataFrame containing property data with feature score columns
            ratings_dict: Dictionary mapping zpid to rating ('yes', 'maybe', 'no')
            
        Returns:
            Tuple of (optimized_weights_dict, optimization_info)
        """
        try:
            # Extract feature data
            feature_matrix, feature_names = self.extract_feature_scores(properties_df)
            
            # Create pairwise comparisons
            comparisons = self.create_pairwise_comparisons(properties_df, ratings_dict)
            
            if len(comparisons) == 0:
                raise ValueError("No valid pairwise comparisons could be created")
            
            # Get current weights in the same order as features
            current_weights = np.array([DEFAULT_FEATURE_WEIGHTS[name] for name in feature_names])
            
            # Set up optimization bounds
            bounds = [self.weight_bounds for _ in range(len(feature_names))]
            
            # Define objective function
            def objective(weights):
                return self.pairwise_ranking_loss(weights, feature_matrix, comparisons, current_weights, feature_names)
            
            # Run optimization
            logger.info(f"Starting optimization with {len(feature_names)} features and {len(comparisons)} comparisons")
            
            result = minimize(
                objective,
                x0=current_weights,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            # Blend optimized weights with current weights for stability
            optimized_weights_array = self.learning_rate * result.x + (1 - self.learning_rate) * current_weights
            
            # Create optimized weights dictionary
            optimized_weights = {}
            for i, feature_name in enumerate(feature_names):
                optimized_weights[feature_name] = float(optimized_weights_array[i])
            
            # Add any missing features from defaults
            for feature_name, default_weight in DEFAULT_FEATURE_WEIGHTS.items():
                if feature_name not in optimized_weights:
                    optimized_weights[feature_name] = default_weight
            
            # Optimization info
            info = {
                'success': result.success,
                'message': result.message,
                'n_iterations': result.nit,
                'final_loss': result.fun,
                'n_comparisons': len(comparisons),
                'n_features': len(feature_names),
                'n_ratings': len([r for r in ratings_dict.values() if r in self.rating_values])
            }
            
            logger.info(f"Optimization {'succeeded' if result.success else 'failed'}: {result.message}")
            logger.info(f"Final loss: {result.fun:.4f}, Iterations: {result.nit}")
            
            return optimized_weights, info
            
        except Exception as e:
            logger.error(f"Weight optimization failed: {e}")
            # Return original weights on failure
            return DEFAULT_FEATURE_WEIGHTS.copy(), {
                'success': False,
                'message': str(e),
                'n_iterations': 0,
                'final_loss': float('inf'),
                'n_comparisons': 0,
                'n_features': 0,
                'n_ratings': len([r for r in ratings_dict.values() if r in self.rating_values])
            }


def scale_weights_to_range(weights: Dict[str, float], target_max: float = 50.0, target_min: float = 0.1) -> Dict[str, float]:
    """
    Scale weights to use full range while preserving relative importance.
    
    Args:
        weights: Dictionary of feature weights
        target_max: Maximum value for scaled weights
        target_min: Minimum value for scaled positive weights
        
    Returns:
        Dictionary of scaled weights
    """
    if not weights:
        return weights.copy()
    
    # Separate positive and negative weights
    positive_weights = {k: v for k, v in weights.items() if v > 0}
    negative_weights = {k: v for k, v in weights.items() if v < 0}
    zero_weights = {k: v for k, v in weights.items() if v == 0}
    
    scaled_weights = {}
    
    # Scale positive weights
    if positive_weights:
        pos_min = min(positive_weights.values())
        pos_max = max(positive_weights.values())
        pos_range = pos_max - pos_min
        
        if pos_range > 0:
            for feature, weight in positive_weights.items():
                # Scale to [target_min, target_max] range
                scaled = target_min + (weight - pos_min) / pos_range * (target_max - target_min)
                scaled_weights[feature] = scaled
        else:
            # All positive weights are the same
            for feature in positive_weights:
                scaled_weights[feature] = target_max / 2
    
    # Scale negative weights (preserve negative sign but scale magnitude)
    if negative_weights:
        neg_min = min(negative_weights.values())  # Most negative
        neg_max = max(negative_weights.values())  # Least negative
        neg_range = neg_max - neg_min
        
        if neg_range > 0:
            for feature, weight in negative_weights.items():
                # Scale magnitude to [target_min, target_max] range, keep negative
                scaled_mag = target_min + (weight - neg_min) / neg_range * (target_max - target_min)
                scaled_weights[feature] = -scaled_mag
        else:
            # All negative weights are the same
            for feature in negative_weights:
                scaled_weights[feature] = -target_max / 2
    
    # Keep zero weights as zero
    for feature in zero_weights:
        scaled_weights[feature] = 0.0
    
    return scaled_weights


def create_weight_url_params(weights: Dict[str, float]) -> str:
    """
    Create URL parameters string from weights dictionary.
    
    Args:
        weights: Dictionary of feature weights
        
    Returns:
        URL parameter string (e.g., "weight_price=20&weight_beds=3")
    """
    params = []
    for feature, weight in weights.items():
        params.append(f"weight_{feature}={weight:.2f}")
    return "&".join(params)
