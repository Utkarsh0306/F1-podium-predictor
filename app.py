from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the re-trained model and encoders
app = Flask(__name__)
try:
    model = joblib.load('f1_model.pkl')
    le_driver = joblib.load('le_driver.pkl')
    le_team = joblib.load('le_team.pkl')
    le_event = joblib.load('le_event.pkl')
    print("Models and encoders loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")
    model = le_driver = le_team = le_event = None

@app.route('/')
def index():
    if le_driver:
        drivers_list = sorted(list(le_driver.classes_))
        teams_list = sorted(list(le_team.classes_))
        events_list = sorted(list(le_event.classes_))
        return render_template('index.html', drivers=drivers_list, teams=teams_list, events=events_list)
    return render_template('index.html', error="Model files not loaded.")

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Error: Model not loaded.", 500

    try:
        # Get the inputs the user provides
        driver = request.form['driver']
        team = request.form['team']
        event = request.form['event']
        is_top_qualifier = int(request.form['top_qualifier'])

        # Set intelligent defaults for the removed inputs
        grid = 2 if is_top_qualifier == 1 else 12
        laptime = 92.0  # Assumed competitive lap time

        # Encode categorical features
        driver_encoded = le_driver.transform([driver])[0]
        team_encoded = le_team.transform([team])[0]
        event_encoded = le_event.transform([event])[0]

        # Create the feature vector with all 6 features
        features = [driver_encoded, team_encoded, event_encoded, grid, is_top_qualifier, laptime]
        X_input = np.array(features).reshape(1, -1)

        # Make prediction
        prediction_result = model.predict(X_input)[0]
        proba_array = model.predict_proba(X_input)[0]
        
        class_1_index = list(model.classes_).index(1)
        confidence = round(proba_array[class_1_index] * 100, 1)

        prediction_text = "PODIUM FINISH" if prediction_result == 1 else "NO PODIUM"

        # Pass the assumed grid position to the results page for clarity
        return render_template('result.html',
                               driver=driver, team=team, event=event, grid=grid, 
                               prediction=prediction_text, confidence=confidence)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('result.html', error="An unexpected error occurred.")

if __name__ == '__main__':
    app.run(debug=True)