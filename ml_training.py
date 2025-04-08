import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from preprocess_data_training import process_nba_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ------------------------------
# Load environment variables and create engine
# ------------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
engine = create_engine(DATABASE_URL, echo=False)

# ------------------------------
# Data Processing
# ------------------------------
# Process the NBA data (this function returns the merged aggregated game data)
merged_games_df = process_nba_data()

# Sort data by date and reset index
merged_games_df.sort_values(by="date", inplace=True)
merged_games_df.reset_index(drop=True, inplace=True)

# Time-based split into train/val/test (80/10/10 split)
df_size = len(merged_games_df)
train_end = int(0.8 * df_size)
val_end = int(0.9 * df_size)
train_df = merged_games_df.iloc[:train_end]
val_df   = merged_games_df.iloc[train_end:val_end]
test_df  = merged_games_df.iloc[val_end:]

# Print dataset sizes and date ranges
print("Full set size:", merged_games_df.shape)
print("Train shape:", train_df.shape)
print("Val shape:", val_df.shape)
print("Test shape:", test_df.shape)

print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
print("Val date range:", val_df["date"].min(), "to", val_df["date"].max())
print("Test date range:", test_df["date"].min(), "to", test_df["date"].max())

# ------------------------------
# Feature and Target Definition
# ------------------------------
target_col = "home_team_won"
feature_cols = [
    "game_postseason",
    "time_in_season",
    # Home Rolling Stats & Context
    "home_team_win_streak", "home_team_elo", "home_opponent_elo",
    "home_pts_rolling15", "home_pie_rolling15",
    "home_field_goals_made_rolling15", "home_field_goals_attempted_rolling15",
    "home_field_goal_percentage_rolling15",
    "home_three_pointers_made_rolling15", "home_three_pointers_attempted_rolling15",
    "home_three_point_percentage_rolling15",
    "home_free_throws_made_rolling15", "home_free_throws_attempted_rolling15",
    "home_free_throw_percentage_rolling15",
    "home_effective_field_goal_percentage_rolling15", "home_true_shooting_percentage_rolling15",
    "home_oreb_rolling15", "home_dreb_rolling15", "home_reb_rolling15",
    "home_offensive_rebound_percentage_rolling15", "home_defensive_rebound_percentage_rolling15",
    "home_rebound_percentage_rolling15", "home_ast_rolling15",
    "home_assist_percentage_rolling15", "home_assist_ratio_rolling15",
    "home_assist_to_turnover_rolling15", "home_stl_rolling15", "home_blk_rolling15",
    "home_turnover_rolling15", "home_turnover_ratio_rolling15", "home_pf_rolling15",
    "home_pace_rolling15", "home_net_rating_rolling15", "home_offensive_rating_rolling15",
    "home_defensive_rating_rolling15", "home_usage_percentage_rolling15",
    "home_absent_players", "home_team_absent_impact", "home_days_since_last_game", "home_travel_distance",
    # Away Rolling Stats & Context
    "away_team_win_streak", "away_team_elo", "away_opponent_elo",
    "away_pts_rolling15", "away_pie_rolling15",
    "away_field_goals_made_rolling15", "away_field_goals_attempted_rolling15",
    "away_field_goal_percentage_rolling15",
    "away_three_pointers_made_rolling15", "away_three_pointers_attempted_rolling15",
    "away_three_point_percentage_rolling15",
    "away_free_throws_made_rolling15", "away_free_throws_attempted_rolling15",
    "away_free_throw_percentage_rolling15",
    "away_effective_field_goal_percentage_rolling15", "away_true_shooting_percentage_rolling15",
    "away_oreb_rolling15", "away_dreb_rolling15", "away_reb_rolling15",
    "away_offensive_rebound_percentage_rolling15", "away_defensive_rebound_percentage_rolling15",
    "away_rebound_percentage_rolling15", "away_ast_rolling15",
    "away_assist_percentage_rolling15", "away_assist_ratio_rolling15",
    "away_assist_to_turnover_rolling15", "away_stl_rolling15", "away_blk_rolling15",
    "away_turnover_rolling15", "away_turnover_ratio_rolling15", "away_pf_rolling15",
    "away_pace_rolling15", "away_net_rating_rolling15", "away_offensive_rating_rolling15",
    "away_defensive_rating_rolling15", "away_usage_percentage_rolling15",
    "away_absent_players", "away_team_absent_impact", "away_days_since_last_game", "away_travel_distance"
]

# Split into features and target for train, validation, and test sets
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_val   = val_df[feature_cols]
y_val   = val_df[target_col]
X_test  = test_df[feature_cols]
y_test  = test_df[target_col]

# ------------------------------
# Model Evaluation Function
# ------------------------------
def evaluate_model(model, X_val, y_val, X_test=None, y_test=None):
    """
    Evaluate a model on validation (and optionally test) data.
    
    Returns:
        dict: Validation metrics.
    """
    def compute_metrics(X, y):
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X)
        else:
            y_prob = None
        auc = roc_auc_score(y, y_prob) if y_prob is not None else None
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}
    
    val_metrics = compute_metrics(X_val, y_val)
    print("Validation Metrics:")
    print(f"Accuracy:  {val_metrics['accuracy']:.3f}")
    print(f"Precision: {val_metrics['precision']:.3f}")
    print(f"Recall:    {val_metrics['recall']:.3f}")
    print(f"F1 Score:  {val_metrics['f1']:.3f}")
    if val_metrics["auc"] is not None:
        print(f"AUC ROC:   {val_metrics['auc']:.3f}")
    else:
        print("AUC ROC:   Not available (model does not support probability estimates)")
    
    if X_test is not None and y_test is not None:
        test_metrics = compute_metrics(X_test, y_test)
        print("\nTest Metrics:")
        print(f"Accuracy:  {test_metrics['accuracy']:.3f}")
        print(f"Precision: {test_metrics['precision']:.3f}")
        print(f"Recall:    {test_metrics['recall']:.3f}")
        print(f"F1 Score:  {test_metrics['f1']:.3f}")
        if test_metrics["auc"] is not None:
            print(f"AUC ROC:   {test_metrics['auc']:.3f}")
        else:
            print("AUC ROC:   Not available (model does not support probability estimates)")
    
    return val_metrics, test_metrics

# ------------------------------
# Grid Search for XGBoost Model
# ------------------------------
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"  # use logloss for evaluation
)

xgb_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.1, 0.01],
    "subsample": [1.0, 0.8],
    "colsample_bytree": [1.0, 0.8]
}

xgb_grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    scoring="accuracy",
    cv=3,
    verbose=1,
    n_jobs=-1
)

xgb_grid_search.fit(X_train, y_train)

print("XGBoost Grid Search Best Params:", xgb_grid_search.best_params_)
print("XGBoost Grid Search Best CV Accuracy:", xgb_grid_search.best_score_)

# Evaluate the best model
best_xgb_grid = xgb_grid_search.best_estimator_
val_metrics, test_metrics = evaluate_model(best_xgb_grid, X_val, y_val, X_test, y_test)

# ------------------------------
# Save the trained model with current date appended to the filename
# ------------------------------
current_date_str = datetime.today().strftime("%Y%m%d")
model_filename = f"best_xgb_model_{current_date_str}.pkl"
joblib.dump(best_xgb_grid, model_filename)
print(f"Trained model saved as {model_filename}")

# ------------------------------
# Record Model Performance in PostgreSQL
# ------------------------------
performance_data = {
    "model_name": "XGBoost Grid Search",
    "date_trained": datetime.today().date(),
    "best_params": str(xgb_grid_search.best_params_),
    "best_cv_accuracy": float(xgb_grid_search.best_score_),
    "validation_accuracy": float(val_metrics["accuracy"]),
    "validation_precision": float(val_metrics["precision"]),
    "validation_recall": float(val_metrics["recall"]),
    "validation_f1": float(val_metrics["f1"]),
    "validation_auc": float(val_metrics["auc"]) if val_metrics["auc"] is not None else None,
    "test_accuracy": float(test_metrics["accuracy"]) if test_metrics is not None else None,
    "test_precision": float(test_metrics["precision"]) if test_metrics is not None else None,
    "test_recall": float(test_metrics["recall"]) if test_metrics is not None else None,
    "test_f1": float(test_metrics["f1"]) if test_metrics is not None else None,
    "test_auc": float(test_metrics["auc"]) if test_metrics["auc"] is not None else None,
    "model_filename": model_filename,
    "n_games": len(X_train)
}

insert_query = text("""
    INSERT INTO model_performance 
    (model_name, date_trained, best_params, best_cv_accuracy, validation_accuracy, validation_precision, validation_recall, validation_f1, validation_auc, test_accuracy, test_precision, test_recall, test_f1, test_auc, model_filename, n_games)
    VALUES (:model_name, :date_trained, :best_params, :best_cv_accuracy, :validation_accuracy, :validation_precision, :validation_recall, :validation_f1, :validation_auc, :test_accuracy, :test_precision, :test_recall, :test_f1, :test_auc, :model_filename, :n_games)
""")

with engine.connect() as conn:
    conn.execute(insert_query, performance_data)
    conn.commit()

print("Model performance recorded in PostgreSQL.")