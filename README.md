# NBA Game Outcome Prediction on Heroku & Streamlit

[Live App →](https://nba-game-predictions-streamlit-49d83933a063.herokuapp.com/)

This project demonstrates how a production-style machine-learning workflow can turn raw NBA data into win-probability forecasts, bookmaker-odds interpretations, and +EV betting signals—all surfaced through a lightweight
Streamlit front-end running on Heroku. It refactors a previous project of mine to be more modular and replaces CSV files with a PostgreSQL database for improved scalability and maintainability.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Tech-Stack](#tech-stack)
4. [Data Pipeline & Database](#data-pipeline--database)
5. [Installation](#installation)
6. [Running the App](#running-the-app)
7. [Further Reading](#further-reading)

## Project Overview

As a data scientist I wanted a concise, end-to-end example that combines:

* **Daily data ingestion** (via *balldontlie* API)  
* **Modular feature-engineering** (rolling 15-game stats, season & all-time Elo,
  rest days, travel distance, injury impact)  
* **Gradient-boosting model** (XGBoost) trained on **5,000 +** historical games  
* **Interpretation of bookmaker odds** → implied win probabilities, vig
  removal, and **expected-value (EV)** calculations  
* **PostgreSQL persistence** and a **Streamlit** dashboard for interactive
  exploration

The application is fully container-free on Heroku: data refresh and model
inference are triggered by **Heroku Scheduler**, and the web UI is served by a
single dyno.

## Key Features

| Category | Description |
|----------|-------------|
| **Predictive Model** | XGBoost classifier with hyper-parameter tuning; outputs a model-derived home-win probability for every upcoming match-up. |
| **Bookmaker Odds Parsing** | American odds from DraftKings & FanDuel are converted to decimal format, then to implied probabilities. The house margin (*vig*) is removed so both sides sum to 100%. |
| **Expected-Value (EV) Calculation** | For each side/book we compute <br> `EV = (p_model × payout) – (1–p_model)` <br>A positive EV indicates theoretical profitability. Cells with the highest EV per game are highlighted. |
| **Automated ETL** | `update_nba_data_api.py` pulls advanced stats, box scores and injuries every morning; `ml_predictions.py` generates fresh predictions. |
| **PostgreSQL Schema** | Central tables: `nba_games`, `nba_game_advanced_stats`, `nba_box_scores`, `nba_player_injuries`, `team_features`, `nba_predictions`, `betting_odds`, `model_performance`. |
| **Streamlit Dashboard** | Four pages: *About*, *Game Predictions* (model vs. books), *Betting Odds* (live odds explorer), *Model Metrics*. |

## Tech-Stack

| Layer | Tools / Packages |
|-------|------------------|
| **Language** | Python 3.11.5 |
| **Data & ML** | `pandas` 2.0, `numpy`, `scikit-learn`, `xgboost`, `joblib` |
| **Geospatial** | `geopy` (travel-distance feature) |
| **Database** | PostgreSQL 14 / `psycopg2` + `SQLAlchemy` |
| **API Client** | `balldontlie` 0.1.6 |
| **Web UI** | `streamlit` 1.44 + `streamlit-option-menu` |
| **Infra** | Heroku dyno, Heroku Postgres, Heroku Scheduler |

Full dependency list in `requirements.txt`.

## Data Pipeline & Database

### Daily Flow

- **[T]** Heroku Scheduler → `python update_nba_data_api.py`  
  &nbsp;&nbsp;&nbsp;&nbsp;↳ Pulls previous day’s games, box scores, injuries, player data  
  &nbsp;&nbsp;&nbsp;&nbsp;↳ Pulls today's games betting odds

- **[T + 1h]** Heroku Scheduler → `python ml_predictions.py`  
  &nbsp;&nbsp;&nbsp;&nbsp;↳ Merges fresh data with historical features  
  &nbsp;&nbsp;&nbsp;&nbsp;↳ Updates Elo & rolling metrics  
  &nbsp;&nbsp;&nbsp;&nbsp;↳ Scores the model and writes to `nba_predictions`  
  &nbsp;&nbsp;&nbsp;&nbsp;↳ Stores feature snapshots in `team_features`

### Database Schema

Some key tables in the PostgreSQL database include:

- **model_performance**: Stores performance metrics (accuracy, F1, AUC, etc.) along with metadata like training date and model details.  
- **nba_game_advanced_stats**: Advanced statistics per game, such as rebound percentages, shooting efficiencies, etc.  
- **nba_box_scores**: Box score information including points, rebounds, assists, etc.  
- **nba_active_players & nba_player_injuries**: Details on active players and any injury-related information.  
- **nba_games & nba_predictions**: Maintains data on games and the corresponding ML predictions.  
- **team_features**: Contains calculations for team metrics such as Elo ratings, win streaks, and time in season.
- **betting_odds**: Contains betting odds for each game from several sportsbooks.

## Installation

### Local Setup

1. **Clone the Repository**  
   `git clone https://github.com/AHijazi11/nba-game-prediction-heroku-streamlit.git`  
   `cd nba-game-prediction-heroku-streamlit`
2. **Set Up a Virtual Environment**  
   On Linux/Mac:  
   `python -m venv venv`  
   `source venv/bin/activate`  
   On Windows:  
   `python -m venv venv`  
   `venv\Scripts\activate`
3. **Install Dependencies**  
   `pip install -r requirements.txt`
4. **Configure Environment Variables**  
   Create a `.env` file in the project root with the following variables:  
   `DATABASE_URL=<Your_PostgreSQL_Database_URL>`  
   `BALLDONTLIE_API_KEY=<Your_Balldontlie_API_Key>`  
   `APP_CONFIG=development`
5. **Database Setup (Optional)**  
   - Ensure PostgreSQL is installed locally.  
   - Create a database and update the `DATABASE_URL` accordingly.  
   - Run any migration or table creation scripts if needed.

### Heroku Setup

1. **Push Your Code to Heroku**  
   `heroku login`  
   `heroku create your-heroku-app`  
   `heroku addons:create heroku-postgresql:hobby-dev`    
   `heroku git:remote -a your-heroku-app-name`  
   `git push heroku main`
3. **Set Environment Variables on Heroku**  
   In the Heroku app’s settings, add the following config vars:  
   - `DATABASE_URL` (from Heroku Postgres)  
   - `BALLDONTLIE_API_KEY`  
   - (Optional) `APP_CONFIG=production`
4. **Configure Heroku Scheduler**  
   Add two scheduled jobs:  
   - **Job 1:** `python update_nba_data_api.py` (runs daily)  
   - **Job 2:** `python ml_predictions.py` (runs daily, one hour after Job 1)

## Running the App

### Running Locally

1. **Update NBA Data** (optional)  
   `python update_nba_data_api.py`
2. **Generate Predictions** (optional)  
   `python ml_predictions.py`
3. **Launch the Streamlit App**  
   `streamlit run app.py`  
   The app will be available at http://localhost:8501

### Deployment on Heroku

Deploy the code to Heroku (as above). The app will automatically use the environment variables and run scheduled jobs.

## Further Reading

Full technical deep-dive (feature engineering, model selection, CV strategy) in the companion repo:  
[https://github.com/AHijazi11/nba-game-outcome-ml](https://github.com/AHijazi11/nba-game-outcome-ml)
