import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import sklearn # For version checking

print(f"Using scikit-learn version: {sklearn.__version__} to regenerate models.")

try:
    # --- Step 1: Load and prepare data (simplified from your notebook) ---
    # Assuming your CSV files are one level up from FlaskIntro
    # Adjust path if they are elsewhere or if your notebook does more complex data fetching
    try:
        df_laps = pd.read_csv('../f1_fastest_laps_2023.csv')
        df_results = pd.read_csv('../f1_driver_results_2023.csv')
    except FileNotFoundError:
        print("ERROR: CSV data files not found. Make sure 'f1_fastest_laps_2023.csv' and 'f1_driver_results_2023.csv' are in the 'F1 predictor' directory (one level above FlaskIntro).")
        exit()

    df_merged = pd.merge(df_laps, df_results, on=['Driver', 'Event'])
    df_merged['Win'] = df_merged['FinishPosition'].apply(lambda x: 1 if x == 1 else 0)
    df_clean = df_merged.dropna().copy()
    df_clean['LapTimeSeconds'] = pd.to_timedelta(df_clean['LapTime']).dt.total_seconds()

    # --- Step 2: Fit Label Encoders ---
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    le_event = LabelEncoder()

    df_clean['DriverEncoded'] = le_driver.fit_transform(df_clean['Driver'])
    df_clean['TeamEncoded'] = le_team.fit_transform(df_clean['Team'])
    df_clean['EventEncoded'] = le_event.fit_transform(df_clean['Event'])

    # --- Step 3: Define Features and Target ---
    features = ['DriverEncoded', 'TeamEncoded', 'EventEncoded', 'GridPosition', 'LapTimeSeconds']
    X = df_clean[features]
    y = df_clean['Win']

    # --- Step 4: Train Model (no train/test split here for simplicity, using all data) ---
    # For a robust model, a train/test split is essential as in your notebook.
    # Here, we are just ensuring the model object is created and saved with the current sklearn version.
    # If you want to use the exact same training data as before, you'd re-implement the split.
    # For this compatibility fix, training on all 'df_clean' data is acceptable for the object structure.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y) # Training on the full cleaned dataset for this regeneration

    # --- Step 5: Save Model and Encoders (will save in FlaskIntro) ---
    joblib.dump(model, 'f1_model.pkl')
    joblib.dump(le_driver, 'le_driver.pkl')
    joblib.dump(le_team, 'le_team.pkl')
    joblib.dump(le_event, 'le_event.pkl')

    print("Successfully regenerated and saved model and encoders in the current directory.")
    print("f1_model.pkl, le_driver.pkl, le_team.pkl, le_event.pkl should now be compatible.")

except Exception as e:
    print(f"An error occurred during model regeneration: {e}")
    import traceback
    traceback.print_exc()