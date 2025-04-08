import os
import psycopg2
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def get_connection():
    """Create and return a connection to the Postgres database using DATABASE_URL."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        st.error("DATABASE_URL environment variable is not set.")
        st.stop()
    conn = psycopg2.connect(db_url)
    return conn


def load_upcoming_predictions(conn, days_ahead: int) -> pd.DataFrame:
    query = f"""
        WITH est_today AS (
            SELECT (CURRENT_TIMESTAMP AT TIME ZONE 'America/New_York')::date AS local_date
        )
        SELECT
            p.game_id,
            p.date,
            p.home_team,
            p.away_team,
            p.predicted_winner
        FROM nba_predictions p
        CROSS JOIN est_today e
        WHERE
            ((p.date::timestamp AT TIME ZONE 'UTC')::date)
            BETWEEN e.local_date
                AND e.local_date + INTERVAL '{days_ahead} DAYS'
        ORDER BY p.date, p.home_team, p.away_team
    """
    df = pd.read_sql(query, conn)
    if 'game_id' in df.columns:
        df.drop(columns='game_id', inplace=True)
    return df

def load_last_week_predictions(conn) -> pd.DataFrame:
    """
    Similar approach for last week's predictions, ensuring the date is converted to an EST-based day.
    """
    query = """
        WITH est_today AS (
            SELECT (CURRENT_TIMESTAMP AT TIME ZONE 'America/New_York')::date AS local_date
        )
        SELECT
            p.game_id,
            p.date,
            p.home_team,
            p.away_team,
            p.predicted_winner,
            g.home_team_score,
            g.visitor_team_score,
            CASE
                WHEN g.home_team_score > g.visitor_team_score THEN p.home_team
                WHEN g.home_team_score < g.visitor_team_score THEN p.away_team
                ELSE 'Tie'
            END AS actual_winner
        FROM nba_predictions p
        JOIN nba_games g
            ON p.game_id = g.id
        CROSS JOIN est_today e
        WHERE
            g.status = 'Final'
            AND ((p.date::timestamp AT TIME ZONE 'UTC')::date)
                BETWEEN e.local_date - INTERVAL '7 DAYS'
                    AND e.local_date
        ORDER BY p.date, p.home_team, p.away_team
    """
    df = pd.read_sql(query, conn)

    # Drop game_id column
    if 'game_id' in df.columns:
        df.drop(columns='game_id', inplace=True)

    # Rename visitor_team_score -> away_team_score
    if 'visitor_team_score' in df.columns:
        df.rename(columns={'visitor_team_score': 'away_team_score'}, inplace=True)

    # Cast score columns to int
    if 'home_team_score' in df.columns:
        df['home_team_score'] = df['home_team_score'].astype(int)
    if 'away_team_score' in df.columns:
        df['away_team_score'] = df['away_team_score'].astype(int)

    return df


def load_model_performance(conn) -> pd.DataFrame:
    query = """
        SELECT
            model_name,
            date_trained,
            best_cv_accuracy,
            validation_accuracy,
            validation_precision,
            validation_recall,
            validation_f1,
            validation_auc,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1,
            test_auc,
            n_games AS trained_on_n_games
        FROM model_performance
        ORDER BY date_trained
    """
    return pd.read_sql(query, conn)


def load_injuries_and_teams(conn) -> pd.DataFrame:
    query = """
        SELECT
            i.player_id,
            i.player_first_name AS first_name,
            i.player_last_name  AS last_name,
            i.player_position   AS position,
            i.status,
            i.return_date,
            COALESCE(p.team_full_name, 'Unknown') AS team
        FROM nba_player_injuries i
        JOIN nba_active_players p
            ON i.player_id = p.id
        ORDER BY p.team_full_name, i.player_last_name
    """
    df = pd.read_sql(query, conn)
    return df

def load_team_features(conn) -> pd.DataFrame:
    """
    Fetch all rows from the team_features table and return as a pandas DataFrame.
    """
    query = """
        SELECT
            game_id,
            date,
            season,
            team_id,
            team_name,
            is_home_game,
            game_postseason,
            game_won,
            win_streak,
            team_elo,
            opponent_elo,
            season_elo,
            opponent_season_elo,
            time_in_season
        FROM team_features
        ORDER BY team_id, date DESC
    """
    df = pd.read_sql(query, conn)
    return df


def load_top_mvp_by_pie(conn) -> pd.DataFrame:
    query_current_season = "SELECT MAX(game_season) AS current_season FROM nba_game_advanced_stats"
    df_max = pd.read_sql(query_current_season, conn)
    if df_max['current_season'].isnull().all():
        return pd.DataFrame(columns=["player_id", "MVP"])
    current_season = int(df_max.iloc[0]['current_season'])
    last_season = current_season - 1

    query_pie = f"""
        SELECT
            player_id,
            player_team_id,
            pie,
            game_season
        FROM nba_game_advanced_stats
        WHERE game_season IN ({last_season}, {current_season})
          AND pie IS NOT NULL
    """
    df_pie = pd.read_sql(query_pie, conn)
    if df_pie.empty:
        return pd.DataFrame(columns=["player_id", "MVP"])

    grouped = df_pie.groupby(["player_team_id", "player_id"], dropna=False)["pie"].mean().reset_index()
    grouped.rename(columns={"pie": "avg_pie"}, inplace=True)

    def rank_by_team(df_sub):
        df_sub = df_sub.sort_values("avg_pie", ascending=False)
        df_sub["rank_within_team"] = range(1, len(df_sub) + 1)
        return df_sub

    ranked = grouped.groupby("player_team_id").apply(rank_by_team).reset_index(drop=True)
    ranked["MVP"] = ranked["rank_within_team"].apply(lambda r: "Yes" if r <= 3 else "No")

    final_mvp = (
        ranked.groupby("player_id")["MVP"]
        .agg(lambda col: "Yes" if "Yes" in col.values else "No")
        .reset_index()
    )

    return final_mvp


# ------------------------------------------------------------------------------
# Highlighting Functions
# ------------------------------------------------------------------------------
def highlight_high_impact(row):
    """Highlight entire row if High Impact == 'Yes' (yellow background & black text)."""
    highlight_style = 'background-color: #ffff99; color: black'
    return [highlight_style if row["High Impact"] == "Yes" else '' for _ in row]

def highlight_prediction_result(row):
    """Highlight faint green if correct, faint red if incorrect (black text)."""
    correct_style   = 'background-color: #ccffcc; color: black'
    incorrect_style = 'background-color: #ffcccc; color: black'
    if row['predicted_winner'] == row['actual_winner']:
        return [correct_style for _ in row]
    else:
        return [incorrect_style for _ in row]

# ------------------------------------------------------------------------------
# Main App
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="NBA Game Outcome Prediction Dashboard", 
    page_icon="🏀",  # basketball emoji
    layout="wide"
)

def main():
    st.title("NBA Game Outcome Prediction Dashboard")

    conn = get_connection()

    # Sidebar Navigation
    page_choice = st.sidebar.radio(
        "Navigation",
        ("Game Predictions", "Model Metrics"),
        index=0
    )

    if page_choice == "Game Predictions":
        st.write("""
        ### NBA Game Predictions – Overview

        - **Upcoming Game Predictions:** Filterable by team and number of days ahead.
        - **Recent Prediction Results:** Compare last week's predicted winners to actual outcomes (green rows for correct predictions, red for incorrect).
        - **Current Player Injuries:** See which players are sidelined and might impact the next games.
        - **Team Season ELO Ratings:** ELO rating for each team in current season.

        ---
        **How it Works:**
        - **Historical NBA data** is gathered (wins/losses, scores, injuries, advanced stats, etc.) from 2020 season to current.
        - Each team's **performance over the last 15 games** is used as a rolling average to capture recent trends (hot streaks, slumps, etc.).
        - **Machine learning model** (trained on around 5000 historical games and increasing) looks for patterns in these features:
        - Whether the team is playing at home or away.
        - Rolling averages of key statistics (points, rebounds, efficiency, etc.).
        - ELO ratings for the season and full data set, win/loss streaks, time-in-season, distance travelled between games, and rest days.
        - Notable player injuries.
        - **Model Training:** Model is trained by feeding it examples of past games, showing which team ultimately won. The model "learns" which factors best predict a victory.
        - **Making Predictions:** Once trained, the model applies the same logic to upcoming matchups, estimating which team is more likely to win.

        **Note:** Even with advanced stats and historical analysis, predictions are not 100% guaranteed. Surprises and upsets happen! The goal is to give a data-informed perspective rather than an absolute certainty.

        ---
        """)

        # -----------------------------
        # Upcoming Predictions
        # -----------------------------
        st.subheader("Upcoming Predictions & Recent Outcomes")

        st.sidebar.subheader("Upcoming Predictions Filters")
        days_ahead = st.sidebar.slider("Days in the Future", min_value=0, max_value=6, value=6)

        df_upcoming = load_upcoming_predictions(conn, days_ahead)
        if df_upcoming.empty:
            st.write("No upcoming predictions found within this EST-based date range.")
        else:
            all_teams = sorted(set(df_upcoming["home_team"].unique()) | set(df_upcoming["away_team"].unique()))
            team_filter = st.sidebar.selectbox("Filter by Team (optional)", options=["All"] + all_teams)

            if team_filter != "All":
                df_upcoming = df_upcoming[
                    (df_upcoming["home_team"] == team_filter) | (df_upcoming["away_team"] == team_filter)
                ]
            if df_upcoming.empty:
                st.write("No upcoming predictions match the filters.")
            else:
                st.dataframe(df_upcoming)

        # -----------------------------
        # Last Week's Predictions vs Actual
        # -----------------------------
        st.subheader("Last Week's Predictions vs. Outcomes")
        df_last_week = load_last_week_predictions(conn)
        if df_last_week.empty:
            st.write("No historical predictions in the last 7 days (EST) or no completed games yet.")
        else:
            styled_last_week = df_last_week.style.apply(highlight_prediction_result, axis=1)
            st.dataframe(styled_last_week)

        # -----------------------------
        # Injured Players & Team ELO side-by-side
        # -----------------------------
        col1, col2 = st.columns(2)

        with col2:
            st.subheader("Injured Players")
            df_injuries = load_injuries_and_teams(conn)
            df_mvp = load_top_mvp_by_pie(conn)
            df_injuries_mvp = pd.merge(df_injuries, df_mvp, on="player_id", how="left")
            df_injuries_mvp["MVP"].fillna("No", inplace=True)

            # Rename "MVP" to "High Impact"
            df_injuries_mvp.rename(columns={"MVP": "High Impact"}, inplace=True)

            cols_to_show = ["first_name", "last_name", "team", "position", "status", "return_date", "High Impact"]
            df_injuries_mvp = df_injuries_mvp[cols_to_show]

            if df_injuries_mvp.empty:
                st.write("No injury data available.")
            else:
                st.write("Rows for high impact players are highlighted.")
                styled_injuries = df_injuries_mvp.style.apply(highlight_high_impact, axis=1)
                st.dataframe(styled_injuries)

        with col1:
            st.subheader("Team Season ELO Ratings")
            team_mini_features_df = load_team_features(conn)
            selected_columns = [
                "season", 
                "team_name",
                "season_elo"
            ]
            if team_mini_features_df.empty:
                st.write("No data found in team_features table.")
            else:
                st.write("Team ELO ratings for current season")
                st.dataframe(team_mini_features_df[selected_columns])
        st.write("""
        ### ELO Rating System:
        - **Initial Rating:** Teams begin with a starting Elo rating of 1500.
        - **Rating Changes:** After each game, ratings adjust based on the game outcome and the rating difference between opponents.
        - **Expected Outcome:** The system computes an expected result for each game, where higher-rated teams are more likely to win.
        - **Points Gained/Lost:**
            - **Expected Win:** A higher-rated teams's win results in a small gain, while the lower-rated team loses a small amount.
            - **Unexpected Win:** An upset win by a lower-rated team results in a larger gain for them and a larger loss for the higher-rated team.
            - **Draw:** Both teams experience slight rating adjustments; the lower-rated team gains a little, and the higher-rated team loses a little.
        - **Self-Correcting Mechanism:** Over time, as more games are played, the Elo rating system adjusts to better reflect a teams's true skill level.
        """)              

    else:
        # -----------------------------
        # Model Metrics
        # -----------------------------
        st.header("Model Performance Metrics")

        df_model = load_model_performance(conn)
        if df_model.empty:
            st.write("No model performance data yet.")
        else:
            # Rename model_name -> model_type
            df_model.rename(columns={"model_name": "model_type"}, inplace=True)

            # Reorder columns so 'trained_on_n_games' appears after 'date_trained'
            desired_order = [
                "id", "model_type", "date_trained", "trained_on_n_games",
                "best_cv_accuracy",
                "validation_accuracy", "validation_precision", "validation_recall", "validation_f1", "validation_auc",
                "test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc",
                "model_filename", "best_params"
            ]
            existing_cols = list(df_model.columns)
            final_cols = [col for col in desired_order if col in existing_cols]
            remaining_cols = [col for col in existing_cols if col not in final_cols]
            final_cols += remaining_cols  # if you want to keep any other columns at the end

            df_model = df_model[final_cols]

            st.dataframe(df_model)

            # Prepare data for line charts:
            df_plot = df_model.copy()
            # Convert date_trained to a real datetime
            df_plot['date_trained'] = pd.to_datetime(df_plot['date_trained'], errors='coerce')
            # Create a string date column so we don't see hours on the x-axis
            df_plot['date_str'] = df_plot['date_trained'].dt.strftime('%Y-%m-%d')
            
            # Use that string column as the index
            df_plot.set_index('date_str', inplace=True, drop=True)

            metrics = ["accuracy", "precision", "recall", "f1", "auc"]
            for metric in metrics:
                val_col = f"validation_{metric}"
                test_col = f"test_{metric}"
                if val_col in df_plot.columns and test_col in df_plot.columns:
                    st.write(f"### {metric.capitalize()}")
                    # Create a small DataFrame with only validation/test columns
                    df_metric = df_plot[[val_col, test_col]].copy()
                    st.line_chart(df_metric)  # x-axis will be date_str, so no hours displayed
                else:
                    st.write(f"No columns found for {metric} (validation/test).")

    conn.close()

if __name__ == "__main__":
    main()