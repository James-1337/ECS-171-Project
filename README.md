This repository contains all materials for the project titled "Predicting Academic Performance from Student Lifestyle Habits", conducted for the ECS 171 course.

Dataset: https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance 
Records: 1,000 students
Features: 15 predictors (e.g., study hours, mental health, diet quality)
Target Variable: Final exam score (continuous, range: 18.4–100)
Data Type: Synthetic but structured to mimic real-world behaviors


Goal: Investigate how lifestyle habits (e.g., study hours, sleep, diet) affect student exam performance. Develop and compare multiple regression-based machine learning models to predict final exam scores.

Problem Statement: Academic success is not solely dependent on intelligence. Lifestyle behaviors—such as consistent sleep, time spent studying, and mental health—can influence outcomes. We aim to build a regression model to predict exam scores based on student habits.


- Develop machine learning models to predict final exam scores from behavioral and demographic variables.
- Compare model performance across several regression-based approaches.
- Interpret model outputs to identify high-impact predictors (e.g., sleep, study time, mental health).
- Evaluate metrics to identify the most suitable model for deployment.
- 
We used four ML models for comparative analysis:
1. Linear Regression (Nico)
2. Polynomial Regression (Madeleine)
3. Random Forest (Shifanaaz)
4. Neural Network (James)

   
Each model was:
- Preprocessed using StandardScaler and appropriate encoding
- Trained with 80/20 train-test split and K-fold cross-validation
Evaluated using:
- R^2 (coefficient of determination)
- MSE (mean squared error)
- MAE, RMSE, and MAPE for error analysis

Best Performing Model: Polynomial Regression (degree = 2)
- R^2 = 0.907
- MSE = 23.72


Outputs: 
- Write-up of the project
- Web Interface
- Demo Video - 5 min
