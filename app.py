from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# --- Improved Model Loading ---
# Load each model file individually to better handle errors
try:
    rf_model = joblib.load('f1_model.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
    le_driver = joblib.load('le_driver.pkl')
    le_team = joblib.load('le_team.pkl')
    le_event = joblib.load('le_event.pkl')
    print("All models and encoders loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR loading model files: {e}")
    # Set them all to None so the app can still run and show an error
    rf_model, xgb_model, le_driver, le_team, le_event = None, None, None, None, None
# --- End of Loading ---


@app.route('/')
def index():
    # This function is now more robust
    drivers_list, teams_list, events_list = [], [], []
    error_message = None

    if le_driver and le_team and le_event:
        drivers_list = sorted(list(le_driver.classes_))
        teams_list = sorted(list(le_team.classes_))
        events_list = sorted(list(le_event.classes_))
    else:
        error_message = "One or more model files could not be loaded. Prediction is unavailable."

    return render_template('index.html',
                           drivers=drivers_list,
                           teams=teams_list,
                           events=events_list,
                           error=error_message)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        driver = request.form['driver']
        team = request.form['team']
        event = request.form['event']
        is_top_qualifier = int(request.form['top_qualifier'])
        model_choice = request.form['model_choice']

        # Set intelligent defaults
        grid = 2 if is_top_qualifier == 1 else 12
        laptime = 92.0

        # Preprocess the data
        driver_encoded = le_driver.transform([driver])[0]
        team_encoded = le_team.transform([team])[0]
        event_encoded = le_event.transform([event])[0]
        features = [driver_encoded, team_encoded, event_encoded, grid, is_top_qualifier, laptime]
        X_input = np.array(features).reshape(1, -1)

        # Select the model based on user's choice
        if model_choice == 'xgb':
            model = xgb_model
            model_name = "XGBoost"
        else:
            model = rf_model
            model_name = "Random Forest"

        # Make prediction
        prediction_result = model.predict(X_input)[0]
        proba_array = model.predict_proba(X_input)[0]
        class_1_index = list(model.classes_).index(1)
        confidence = round(proba_array[class_1_index] * 100, 1)
        prediction_text = "PODIUM FINISH" if prediction_result == 1 else "NO PODIUM"

        # Get feature importances
        feature_names = ['Driver', 'Team', 'Event', 'Grid Position', 'Top 3 Qualifier', 'Lap Time']
        importances = model.feature_importances_
        feature_importance_data = {name: round(float(imp) * 100, 1) for name, imp in zip(feature_names, importances)}

        return render_template('result.html',
                               driver=driver, team=team, event=event, grid=grid,
                               prediction=prediction_text, confidence=confidence,
                               feature_importances=feature_importance_data,
                               model_used=model_name)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('result.html', error="An unexpected error occurred.")


if __name__ == '__main__':
    app.run(debug=True)
