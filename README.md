# NBA Game Outcome Prediction on Heroku & Streamlit

[Streamlit App](https://nba-game-predictions-streamlit-49d83933a063.herokuapp.com/)

The NBA Game Prediction Project leverages machine learning to predict the outcomes of NBA games—showcasing how data-driven insights can be applied to something many people enjoy. It refactors a previous project of mine to be more modular and replaces CSV files with a PostgreSQL database for improved scalability and maintainability. The app, along with its data ingestion and prediction scripts, is deployed on Heroku.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack & Dependencies](#tech-stack--dependencies)
- [Data Pipeline & Database](#data-pipeline--database)
- [Installation & Setup](#installation--setup)
- [Usage & Deployment](#usage--deployment)
- [Contributing](#contributing)
- [License](#license)
- [Further Reading](#further-reading)

## Overview

The goal of this project is to leverage machine learning to predict NBA game outcomes, thereby demonstrating its usefulness in an area that excites both NBA fans and machine learning enthusiasts. The app provides updated predictions, advanced analytics, and interactive visualizations through a clean Streamlit interface.

Key components include:
- **Daily Data Updates:** Data is fetched daily from external sources using the balldontlie API to ensure the latest game stats.
- **Feature Engineering & Predictions:** Modular functions in `preprocess_data_prediction.py` and `preprocess_data_training.py` handle data processing. These functions are imported into `ml_predictions.py` and `ml_training.py` to compute rolling averages, update team Elo ratings, and generate predictions.
- **Database Integration:** A PostgreSQL database is used to store and manage historical and real-time data—no more CSV files.
- **Modular Architecture:** The project has been refactored to improve maintainability and ease of extension.

## Features

- **Predictive Analytics:**  
  Uses machine learning models (including XGBoost and scikit-learn algorithms) to predict game outcomes.
- **Automated Data Updates:**  
  - `update_nba_data_api.py` runs daily via Heroku Scheduler to retrieve advanced stats, box scores, player injuries, and game data.
  - `ml_predictions.py` runs after update_nba_data_api.py to process features and generate predictions.
- **PostgreSQL Database:**  
  Utilizes a PostgreSQL database with several key tables, including:
  - `model_performance`
  - `nba_game_advanced_stats`
  - `nba_box_scores`
  - `nba_active_players`
  - `nba_player_injuries`
  - `nba_games`
  - `nba_predictions`
  - `team_features`
- **Companion Repository:**  
  For further details on data analysis, feature engineering, and model training, please refer to the [NBA Game Outcome ML Repo](https://github.com/AHijazi11/nba-game-outcome-ml).

## Tech Stack & Dependencies

- **Python Version:** 3.11.5  
- **Streamlit Version:** 1.44.0  
- **Dependencies:**
  - balldontlie==0.1.6
  - geopy==2.4.1
  - joblib==1.4.2
  - numpy==1.24.3
  - pandas==2.0.3
  - psycopg2==2.9.10
  - python-dotenv==1.1.0
  - scikit_learn==1.3.0
  - SQLAlchemy==1.4.39
  - xgboost==3.0.0

The complete list is available in the `requirements.txt` file.

## Data Pipeline & Database

### Data Pipeline

- **Data Ingestion:**  
  The `update_nba_data_api.py` script runs daily (via Heroku Scheduler) to fetch updated NBA data (advanced stats, box scores, injuries, etc.) using the balldontlie API.
- **Feature Engineering & Prediction:**  
  Modular functions in `preprocess_data_prediction.py` and `preprocess_data_training.py` process the NBA data. The `ml_predictions.py` script utilizes these functions to calculate features (e.g., 15-game rolling averages, Elo ratings) before generating predictions.

### Database Schema

Some key tables in the PostgreSQL database include:

- **model_performance**: Stores performance metrics (accuracy, F1, AUC, etc.) along with metadata like training date and model details.  
- **nba_game_advanced_stats**: Advanced statistics per game, such as rebound percentages, shooting efficiencies, etc.  
- **nba_box_scores**: Box score information including points, rebounds, assists, etc.  
- **nba_active_players & nba_player_injuries**: Details on active players and any injury-related information.  
- **nba_games & nba_predictions**: Maintains data on games and the corresponding ML predictions.  
- **team_features**: Contains calculations for team metrics such as Elo ratings, win streaks, and time in season.

## Installation & Setup

### Local Setup

1. **Clone the Repository**  
   `git clone https://github.com/YourUsername/nba-game-prediction-streamlit.git`  
   `cd nba-game-prediction-streamlit`
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
   `heroku git:remote -a your-heroku-app-name`  
   `git push heroku main`
2. **Set Environment Variables on Heroku**  
   In the Heroku app’s settings, add the following config vars:  
   - `DATABASE_URL` (from Heroku Postgres)  
   - `BALLDONTLIE_API_KEY`  
   - (Optional) `APP_CONFIG=production`
3. **Configure Heroku Scheduler**  
   Add two scheduled jobs:  
   - **Job 1:** `python update_nba_data_api.py` (runs daily)  
   - **Job 2:** `python ml_predictions.py` (runs daily, one hour after Job 1)

## Usage & Deployment

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

For more details on the data analysis, feature engineering, and machine learning models used in this project, please refer to the companion repository:  
[https://github.com/AHijazi11/nba-game-outcome-ml](https://github.com/AHijazi11/nba-game-outcome-ml)