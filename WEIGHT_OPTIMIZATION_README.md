## Weight Optimization for Real Estate Scoring

### Overview

Learns feature weights from your yes/maybe/no ratings using pairwise ranking. A single aggressiveness knob controls regularization, blending with current weights, and ranking margin. Integrated with the Flask app via `POST /optimize_weights`.

### What it does
- **Extract features**: Uses columns ending with `_score`; filters to features present in `DEFAULT_FEATURE_WEIGHTS`.
- **Build preferences**: Maps ratings `yes=2`, `maybe=1`, `no=0`; creates pairwise comparisons where ratings differ.
- **Optimize**: Minimizes hinge-style pairwise ranking loss with L2 regularization using L-BFGS-B and bounded weights.
- **Blend for stability**: Final weights are a blend of optimized and current weights.
- **Output**: Returns an updated weight dict and optimization info; Flask route also returns URL params and re-scored properties.

### Python API
```python
from weight_optimizer import WeightOptimizer

optimizer = WeightOptimizer(
    aggressiveness=0.4,      # single control knob (0.1 conservative … 0.9 aggressive)
    weight_bounds=(-50, 50), # global bounds
    min_ratings=10           # require enough signal
)

optimized_weights, info = optimizer.optimize_weights(properties_df, ratings_dict)

if info['success']:
    print(info['n_iterations'], info['n_ratings'])
else:
    print('Failed:', info['message'])
```

### Aggressiveness (single control)
- **Recommended**: 0.4–0.6 (default: 0.4)
- Internally derives:
  - **L2 regularization**: stronger when conservative, weaker when aggressive
  - **Blend factor**: how much to move from current to optimized weights
  - **Ranking margin**: separation between rating levels

At default 0.4, the behavior is moderately conservative and stable.

### Negative feature handling
For features where higher values are worse (`road_exposure`, `avg_risk_severity`, `avg_crime_severity`, `avg_school_distance`, `drive_time`):
- If weight > 0: use inverted scores `(100 - feature_score)`
- If weight < 0: use raw scores with absolute weight

### Flask endpoint
`POST /optimize_weights`
- **Auth**: `password` must match server config
- **Optional baseline**: `weight_<feature>` form fields override starting weights
- **Filters**:
  - `financing_filter` (default: `eligible`)
  - `rating_filter` (default display filter: `yes`, `maybe`, `blank`)
- **Response**:
  - `success`: boolean
  - `optimized_weights`: dict
  - `weight_url_params`: query-string of weights
  - `optimization_info`: `{success, message, n_iterations, final_loss, n_comparisons, n_features, n_ratings}`
  - `properties`: re-scored list (present only when success)

Note: The route does not scale optimized weights before using them (scaling would distort the learned solution).

### Defaults and constraints
- `aggressiveness = 0.4`
- `weight_bounds = (-50, 50)`
- `min_ratings = 10`

### Error handling and performance
- Returns original weights with an error message if: insufficient ratings, no valid features, no pairwise comparisons, or optimization failure.
- Designed to run quickly on typical datasets; uses stable 0–100 feature scoring.

### Math (brief)
For each differing-rated pair (i, j):
```
loss += max(0, -preference_diff * (score_i - score_j) + margin)
```
Final weights are blended: `final = blend * optimized + (1 - blend) * current`.

### Dependencies
`scipy`, `numpy`, `pandas`
