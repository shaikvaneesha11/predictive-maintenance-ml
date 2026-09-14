# Predictive Maintenance for Industrial Equipment

Machine learning project that predicts equipment failures before they occur, using historical sensor data and operational logs — reducing downtime and maintenance costs.

## Business problem
Unexpected machinery failures cause significant losses in industrial settings. This project builds a predictive maintenance (PdM) model that flags equipment likely to fail soon, so maintenance teams can act proactively instead of reactively.

## Dataset
Kaggle - Predictive Maintenance Dataset, containing:
- Sensor readings (temperature, pressure, vibration)
- Equipment ID
- Failure types
- Time since last maintenance

## Approach
1. **Data cleaning** — handled missing values, label-encoded categorical columns, normalized sensor readings.
2. **Feature engineering** — rolling mean/std of sensor readings, lag features to capture prior sensor states, failure label as the target variable.
3. **Modeling** — trained a Random Forest Classifier to predict failure events.
4. **Evaluation** — assessed with classification report and confusion matrix.

## Results
- **Accuracy: ~95%**
- High precision and recall, with very few false alarms
- Confusion matrix shows most predictions correctly on the diagonal (few false positives/negatives)

## Tools used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Author
Shaik Vaneesha Begum — CSE (Data Science), KKR & KSR Institute of Technology and Sciences
