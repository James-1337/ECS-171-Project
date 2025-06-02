from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define SimpleNet
class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load preprocessor
preprocessor = joblib.load('preprocessor.joblib')

# Load model
INPUT_DIM = 25
model = SimpleNet(INPUT_DIM)
model.load_state_dict(torch.load("simplenet_model.pth", map_location=torch.device('cpu')))
model.eval()

app = Flask(__name__)

# Define feature names and types
num_cols = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours', 'attendance_percentage', 'sleep_hours', 'exercise_frequency', 'mental_health_rating']
cat_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality', 'extracurricular_participation']
feature_names = num_cols + cat_cols

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    classification = None

    if request.method == 'POST':
        try:
            # Collect form inputs using exact feature names
            input_dict = {
                'age': [request.form['age']],
                'study_hours_per_day': [request.form['study_hours_per_day']],
                'social_media_hours': [request.form['social_media_hours']],
                'netflix_hours': [request.form['netflix_hours']],
                'attendance_percentage': [request.form['attendance_percentage']],
                'sleep_hours': [request.form['sleep_hours']],
                'exercise_frequency': [request.form['exercise_frequency']],
                'mental_health_rating': [request.form['mental_health_rating']],
                'gender': [request.form['gender']],
                'part_time_job': [request.form['part_time_job']],
                'diet_quality': [request.form['diet_quality']],
                'parental_education_level': [request.form['parental_education_level'] or 'none'],
                'internet_quality': [request.form['internet_quality']],
                'extracurricular_participation': [request.form['extracurricular_participation']]
            }

            # Convert to DataFrame
            input_df = pd.DataFrame(input_dict)

            # Convert numerical inputs to float
            for col in num_cols:
                input_df[col] = pd.to_numeric(input_df[col], errors='raise')

            # Apply preprocessing
            input_processed = preprocessor.transform(input_df)

            # Convert to tensor
            input_tensor = torch.tensor(input_processed, dtype=torch.float32)

            # Make prediction
            with torch.no_grad():
                score = model(input_tensor).item()
            prediction = round(score, 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction, classification=classification)

if __name__ == '__main__':
    app.run(debug=True)