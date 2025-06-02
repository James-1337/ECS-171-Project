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
        # Request study hours and mental health rating and pad everything else with 0
        study_hours = float(request.form["study_hours"])
        mental_rating = float(request.form["mental_rating"])
        x = np.array([[study_hours, 0, 0, 0, 0, 0, mental_rating]])
        
        x_scaled = scaler.transform(x)
        x_poly = poly.transform(x_scaled)
        y_pred = model.predict(x_poly)[0]
        prediction = round(y_pred, 2)
    return render_template_string(TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
