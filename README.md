# Rainfall Prediction Classifier

In this project, I built a **Rainfall Prediction Classifier** using a real-world weather dataset from the **Australian Government Bureau of Meteorology**.  
The dataset was preprocessed, analyzed, and used to train multiple **machine learning classification models** to predict **whether it will rain tomorrow or not**.

This project focuses on **feature engineering, pipeline building, hyperparameter tuning, and model evaluation** using **Scikit-Learn**.

---

## **Dataset Information**

The dataset was sourced from the **Australian Government's Bureau of Meteorology** and obtained via **Kaggle**:  
[Weather Dataset on Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

The dataset contains **daily weather observations** across multiple Australian locations from **2008 to 2017**.

| **Field**        | **Description**                                                        | **Unit**           | **Type**  |
|-------------------|------------------------------------------------------------------------|---------------------|-----------|
| Date             | Date of observation                                                    | YYYY-MM-DD          | object    |
| Location         | Observation location                                                   | City/Region         | object    |
| MinTemp          | Minimum temperature                                                    | °C                  | float     |
| MaxTemp          | Maximum temperature                                                    | °C                  | float     |
| Rainfall        | Amount of rainfall                                                     | mm                  | float     |
| Evaporation     | Amount of evaporation                                                 | mm                  | float     |
| Sunshine       | Amount of bright sunshine                                             | Hours               | float     |
| WindGustDir    | Direction of strongest gust                                           | Compass Points      | object    |
| WindGustSpeed | Speed of strongest gust                                              | km/h                | float     |
| WindDir9am     | Wind direction averaged prior to 9am                                | Compass Points      | object    |
| WindDir3pm     | Wind direction averaged prior to 3pm                                | Compass Points      | object    |
| WindSpeed9am  | Wind speed averaged prior to 9am                                    | km/h                | float     |
| WindSpeed3pm  | Wind speed averaged prior to 3pm                                    | km/h                | float     |
| Humidity9am   | Humidity at 9am                                                     | %                   | float     |
| Humidity3pm   | Humidity at 3pm                                                     | %                   | float     |
| Pressure9am   | Atmospheric pressure at 9am                                        | hPa                 | float     |
| Pressure3pm   | Atmospheric pressure at 3pm                                        | hPa                 | float     |
| Cloud9am      | Fraction of sky obscured by cloud at 9am                           | Eighths             | float     |
| Cloud3pm      | Fraction of sky obscured by cloud at 3pm                           | Eighths             | float     |
| Temp9am       | Temperature at 9am                                                | °C                  | float     |
| Temp3pm       | Temperature at 3pm                                                | °C                  | float     |
| RainToday     | Whether it rained today (Yes/No)                                  | Categorical         | object    |
| RainTomorrow  | **Target Variable** → Will it rain tomorrow?                      | Categorical         | object    |

---

## **Objectives**

After completing this project, I was able to:

- Explore and preprocess a **real-world weather dataset**.
- Perform **feature engineering** and handle missing values.
- Build an **end-to-end classifier pipeline** using **Scikit-Learn**.
- Optimize the pipeline using **GridSearchCV** and **cross-validation**.
- Evaluate the models using various **classification metrics**.
- Implement multiple classifiers and tune their hyperparameters.

---

## **Project Overview**

This project involves building a **rainfall prediction classifier** step by step:

### **Key Steps**
1. **Importing Required Libraries**  
   - NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn.
   
2. **Loading the Dataset**  
   Read the CSV file, inspected its structure, and checked for missing values.

3. **Data Preprocessing**  
   - Handling missing data with **imputation techniques**.
   - Encoding categorical variables using **OneHotEncoder**.
   - Scaling numerical features with **StandardScaler**.

4. **Exploratory Data Analysis (EDA)**  
   - Understanding rainfall patterns across regions.
   - Correlation analysis between weather features.
   - Visualizing rainfall trends using Matplotlib and Seaborn.

5. **Building the Classifier Pipeline**  
   - Created a pipeline that integrates preprocessing and modeling.
   - Used **Logistic Regression**, **Random Forest**, and **XGBoost**.

6. **Hyperparameter Tuning**  
   - Implemented **GridSearchCV** to find the best model parameters.
   - Used **cross-validation** for performance evaluation.

7. **Model Evaluation**  
   Evaluated models using:
   - Accuracy Score
   - Precision, Recall, and F1-Score
   - Confusion Matrix
   - ROC Curve & AUC Score

8. **Implementing a Different Classifier**  
   Replaced the classifier in the pipeline with a **Random Forest** and **XGBoost** to compare results.

---

## **Technologies Used**
- Python 3
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib / Seaborn (for visualization)
- XGBoost (optional, for better accuracy)

---

