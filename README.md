# Flight Arrival Delay Prediction Project

## Overview
This project aims to predict flight arrival delays using various regression algorithms. The goal is to apply multiple machine learning models and identify the best-performing model for this task. The models used in this project include:

- CatBoost Regressor
- XGBoost Regressor
- LightGBM Regressor
- Support Vector Machine (SVM) Regressor
- Lasso Regressor
- Ridge Regressor
- Decision Tree Regressor
- Random Forest Regressor

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction
Predicting flight arrival delays is a critical task for airlines and passengers. This project explores the application of several advanced machine learning algorithms to create an accurate prediction model for flight arrival delays. By comparing different models, I aim to find the best approach for this problem.

## Project Structure
```
flight_arrival_delay_prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
├── notebooks/
│   ├── EDA_and_feature_engineering.ipynb
│   ├── Delays_Modeling_popular_models.ipynb
│   ├── delays_modeling.ipynb
│   ├── lightgbm_parameter_tuning.ipynb
│   ├── cateboost_parameter_tuning.ipynb
│   ├── xgboost_parameter_tuning_and_voting_regressor.ipynb
│   ├── best_models_voting_regressor.ipynb
│   ├── best_single_model_lightgbm.ipynb
├── README.md
├── requirements.txt
```


## Dependencies
To install the required dependencies, use the following command:
```bash
pip install -r requirements.txt
```

The main libraries used in this project include:

    pandas
    numpy
    catboost
    xgboost
    lightgbm
    scikit-learn
    matplotlib
    seaborn

## Data

The data folder contains the training and test datasets (train.csv and test.csv). The data preprocessing steps are outlined in the EDA_and_feature_engineering.ipynb notebook.
Preprocessing

Data preprocessing involves cleaning the dataset, handling missing values, feature engineering, and scaling the features. These steps are crucial for preparing the data for model training. The detailed steps are in the EDA_and_feature_engineering.ipynb notebook too.
Model Training

The model training process, including hyperparameter tuning, is documented in the Delays_Modeling_popular_models.ipynb notebook. I use cross-validation to ensure robust performance estimation. The training includes:

    Splitting the data into training and validation sets.
    Training models using CatBoost, XGBoost, LightGBM, SVM, Lasso, Ridge, Decision Tree, and Random Forest regressors.
    Hyperparameter tuning using grid search and random search techniques.

## valuation

The model evaluation, including cross-validation and Mean Absolute Percentage Error (MAPE) calculation, is detailed in the xgboost_parameter_tuning_and_voting_regressor.ipynb notebook. I compare the performance of different models to identify the best one.

Results

After fine tuning every model and train them with train data, I found the best single model is LightGBM regressor and the best Voting Regressor model is combination of LightGBM, Catboost, XGBoost regressor with reaching 6.46 % MAPE

## Usage

To preprocess the data, open and run the EDA_and_feature_engineering.ipynb notebook.

To train the models, open and run the Delays_Modeling_popular_models.ipynb notebook.

To evaluate the models and compare their performance, open and run the notebook named parameter_tuning.ipynb.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.



