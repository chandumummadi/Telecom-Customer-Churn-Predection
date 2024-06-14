# Customer Churn Prediction

## Overview
This project aims to predict customer churn using machine learning techniques. The dataset used is from a telecommunications company and includes various features related to customer demographics, account information, and services subscribed to. The project involves data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.

## Project Structure
- `final_project_ML_customerChurn.ipynb`: Jupyter notebook containing the complete workflow for customer churn prediction.
- `Telco_customer_churn.xlsx`: Dataset used for the analysis and modeling.

## Getting Started

### Prerequisites
- Python 3.6 or higher
- Required Python libraries:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn
  - xgboost
  - wordcloud

### Installation
To install the necessary libraries, run:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost wordcloud
```

### Dataset
The dataset used in this project is `Telco_customer_churn.xlsx`, which contains information about the customers and their subscription details.

## Data Preprocessing
- **Missing Values:** Handled missing values in the 'Total Charges' column by converting it to numeric and imputing with the median.
- **Feature Scaling:** Applied MinMax scaling to 'Tenure Months' and 'Monthly Charges'.
- **Categorical Encoding:** Used one-hot encoding for categorical features.

## Exploratory Data Analysis (EDA)
- **Visualization:** Utilized count plots, KDE plots, and pie charts to explore the distribution of various features and their relationship with churn.
- **Insights:**
  - Analyzed the distribution of churn across different features like gender, seniority, partner status, dependents, contract types, and payment methods.
  - Identified specific subsets of customers with high churn rates, such as those with poor activation.

## Feature Engineering
- Removed records with poor activation to focus on relevant data.
- Selected and transformed necessary features for modeling.

## Model Training and Evaluation
### Initial Model
- **Model:** XGBoost Classifier
- **Performance:** 
  - Training Accuracy: Initial accuracy on training data.
  - Test Accuracy: Initial accuracy on test data.

### Hyperparameter Tuning
- **Method:** GridSearchCV
- **Hyperparameters Tuned:** max_depth, learning_rate, n_estimators, subsample
- **Best Model:** Trained with optimal hyperparameters.

### Final Model Evaluation
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix:** Visualized to show the performance of the model.

## Results
- **Training Accuracy:** Detailed in the notebook.
- **Test Accuracy:** Detailed in the notebook.
- **Precision:** Detailed in the notebook.
- **Recall:** Detailed in the notebook.
- **F1-Score:** Detailed in the notebook.

## Conclusion
This project demonstrates a comprehensive approach to customer churn prediction using machine learning. The final model, an XGBoost classifier, achieved satisfactory performance with accuracy, precision, recall, and F1-score metrics.
