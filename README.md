# F1 Podium Predictor 🏎️🏁

**Live Application:** [https://f1-podium-predictor-st3u.onrender.com/](https://f1-podium-predictor-st3u.onrender.com/)

This project is a full-stack web application that leverages machine learning to predict whether a Formula 1 driver will finish on the podium. The entire process, from raw data collection via the fastf1 API to a deployed, interactive web app, was handled. The model was initially trained to predict a win but was later refined to predict a podium finish to solve the severe class imbalance problem, resulting in a more robust and reliable model.

The application allows users to select between two different models (Random Forest and XGBoost) and provides model explainability by visualizing the most important features that contributed to the prediction.

<img width="859" height="855" alt="image" src="https://github.com/user-attachments/assets/516bd22c-e50b-43c9-b74c-bdea3d63c642" />


---

## Project Summary

This project is a full-stack web application that leverages a machine learning model to predict whether a Formula 1 driver will finish on the podium. The entire process, from raw data collection via the `fastf1` API to a deployed, interactive web app, was handled. The model was initially trained to predict a win but was later refined to predict a podium finish to solve the severe class imbalance problem, resulting in a more robust and reliable model.

---

## Key Features

* **Data Sourcing:** Programmatically collected and cleaned multiple seasons of F1 race data using the `fastf1` library.
* **Feature Engineering:** Created new predictive features, such as `is_top_3_qualifier`, to improve model accuracy.
* **Machine Learning Model:** Trained and compared multiple models, including a Random Forest baseline and a more advanced XGBoost model, to find the best performance.
* **Model Explainability:** Integrated feature importance charts to interpret and visualize what factors drive each model's predictions.
* **Backend:** Developed a RESTful API using Python and Flask to serve model predictions.
* **Frontend:** Designed a responsive and modern user interface with HTML and CSS.
* **Deployment:** Packaged the application with Gunicorn and deployed it live to the cloud using Render.

---

## Technology Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn, XGBoost, Pandas, NumPy
* **Server:** Gunicorn
* **Frontend:** HTML, CSS
* **Deployment:** Render, Git

---

## Setup & Installation

To run this project locally:

1.  Clone the repository:
    ```bash
    git clone https://github.com/Utkarsh0306/F1-podium-predictor.git
    cd F1-podium-predictor
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the Flask application:
    ```bash
    python app.py
    ```
5.  Open your browser and navigate to `http://127.0.0.1:5000`.
