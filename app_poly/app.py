from flask import Flask, request, render_template_string
import numpy as np
import joblib

app = Flask(__name__)

# Loading models and template
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
poly = joblib.load("poly.pkl")
TEMPLATE = open("templates/index.html").read()

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        # Map diet quality to numerical values
        diet_quality_map = {"Poor": 0, "Fair": 1, "Good": 2}

        # Request every feature
        features = [
            float(request.form["study_hours_per_day"]),
            float(request.form["social_media_hours"]),
            float(request.form["netflix_hours"]),
            float(request.form["attendance_percentage"]),
            float(request.form["sleep_hours"]),
            diet_quality_map[request.form["diet_quality"]],
            float(request.form["exercise_frequency"]),
            float(request.form["mental_health_rating"])
        ]
        x = np.array([features])

        x_scaled = scaler.transform(x)
        x_poly = poly.transform(x_scaled)
        y_pred = model.predict(x_poly)[0]
        prediction = round(y_pred, 2)
    return render_template_string(TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
