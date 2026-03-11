# 🏡 AI Property Hunter

*By Shawn Strasser*

**🔗 Live Demo:** [house.up.railway.app](https://house.up.railway.app)

---

A smart property analysis tool that uses AI to evaluate real estate listings the way I would - checking everything from kitchen quality to crime rates. It scrapes Zillow, analyzes photos with Google's Gemini AI, and scores properties based on 25+ factors I care about. Plus, there's a mobile-friendly website where my wife and I can browse through properties and save our favorites.

---

## 🎯 The Cool Parts

**AI that sees what you see**
- Google Gemini 2.5 Pro turns photos + text into 25+ human-centered features
- Clear ratings for kitchen, baths, privacy, views, road exposure, and more

**Scoring that blends real-world data**
- FBI crime stats, drive time to grocery, and school ratings
- Smart normalization + customizable weights → one meaningful total score

**Built to actually use**
- Mobile web app to browse, rate (Yes/Maybe/No), and add notes
- Production-ready, cloud-backed ratings; simple, private, and fast

---

## 🛠️ Data Pipeline


```
┌─────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Zillow  │───▶│ Playwright       │───▶│ Gemini AI       │
│ Listings│    │ Scraper          │    │ Analysis        │
└─────────┘    └──────────────────┘    └─────────────────┘
                                                │
                                                ▼
┌─────────────┐                        ┌──────────────┐
│ Crime Data  │─────────────── ───────▶│ DuckDB       │
└─────────────┘                        │ Scoring      │
                                       │              │
┌─────────────┐    ┌──────────────┐───▶│              │
│ AI Results  │───▶│ Maps API     │    │              │
│ (Addresses) │    │ (Grocery     │    │              │
└─────────────┘    │  Stores)     │    └──────────────┘
                   └──────────────┘             │
                                                ▼
                                       ┌─────────────────┐
                                       │ Flask Web App   │
                                       └─────────────────┘
                                                │
                                                ▼
                                       ┌─────────────────┐
                                       │ Railway Deploy  │
                                       └─────────────────┘
```


---

---

## 📁 Project Structure

```
House/
├── flask_app.py           # Flask app (UI/API, SQLite Cloud ratings)
├── live_scraper.py        # Zillow scraper + Gemini feature extraction
├── feature_extraction.py  # Gemini prompts/tools + features table schema
├── scoring.py             # Scoring SQL and helpers
├── nearest_grocery.py     # Google Maps: nearest grocery + drive time → grocery table
├── config.py              # Weights, scoring params, app config, emojis
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── settings.html
│   └── error.html
├── crime.ipynb            # Populate crime data (creates crime table)
├── requirements.txt
├── railway.toml
└── property_data.db       # DuckDB file (created at runtime by scripts)
```


---

## 🚀 How to Use

1) Update the Zillow search URL
- Edit `scripts/live_scraper.py` and change `search_url` to your target area/filters.

2) Set environment variables
- `GEMINI_API_KEY` (Google Generative AI)
- `MAPS_API_KEY` (Google Maps Platform)
- `SECRET_KEY` and `APP_PASSWORD` (Flask app)
- `RATINGS_DB_URL` (SQLite Cloud URL for ratings/notes)

3) Run the scraper
- `python live_scraper.py` or `python scripts/live_scraper.py` (creates/updates `property_data.db` with listings + AI features)

4) Add crime data
- Open `notebooks/crime.ipynb` and run it to populate the `crime` table.

5) Add grocery stores
- `python nearest_grocery.py` or `python scripts/nearest_grocery.py` (uses `MAPS_API_KEY` to fill the `grocery` table with nearest store + drive time)

6) Deploy to Railway
- Push the repo, connect on Railway, set the env vars above, and deploy
