# Credit Card Approval Prediction
## Overview
This project aims to predict credit card approval decisions based on various features related to credit history, financial information, and personal details of applicants. The goal is to build a machine learning model that can accurately predict whether a credit card application will be approved or rejected by the issuing authority.

## Dataset
The dataset used in this project contains historical credit card application data, including both approved and rejected applications. It includes features such as applicant's age, income, credit score, employment status, debt-to-income ratio, and more.

The dataset is stored in the 'data' directory and is split into two files: 'credit_card_applications.csv' and 'credit_card_labels.csv'. The 'credit_card_applications.csv' file contains the feature data, while the 'credit_card_labels.csv' file contains the corresponding labels (0 for rejected and 1 for approved).

## Requirements
To run the code and reproduce the results, you need the dependencies in 'requirements.txt' file.

## Installation
You can install the required packages using the following command:

<pre>
pip install -r requirements.txt
</pre>

This command needs to be run in the terminal after navigating to the directory containing the 'requirements.txt' file.

## Data Preprocessing
Data preprocessing involved handling missing values, converting categorical variables into numerical form, and balancing the target variable using SMOTE technique due to our imbalnce data as following picture.

![Data Imbalance](https://github.com/tutalae/creditcard_approval_prediction/blob/main/results/imbalance%20data.png)

## Experiments
During the experimentation phase, several models including RandomForest, ElasticNet, DecisionTree, GradientBoosting and XGBoost were evaluated using different hyperparameters. MLflow was used to track the experiments. Performance was evaluated based on RMSE, MAE, and R2 metrics.

## Running the Scripts
All the code and experiments are contained in the Jupyter notebook 'simple_exp_tracking_mlflow.ipynb'. You can run this notebook using Jupyter Notebook or Jupyter Lab.

To start the notebook, navigate to the project directory in the terminal and type 'jupyter notebook' or 'jupyter lab'. This will start the Jupyter server and open a tab in your web browser where you can select and run the notebook.

## MLflow Tracking
Experiments and model performance metrics were logged using MLflow. To view the experiment results, start the MLflow server using the command 

<pre>
mlflow ui --backend-store-uri sqlite:///mlflow.db
</pre>

from the project directory. This will start the MLflow server and provide a local URL that you can open in your web browser to view the experiment results.

## Results and Visualizations
The results of the model evaluations and some visualizations can be found in the 'results' folder. They provide insights into the performance of the models and the importance of different features.

## Folder Structure and File Descriptions
- data: This folder contains the dataset files.
- simple_exp_tracking_mlflow.ipynb: This is the main Jupyter notebook that contains all the code.
- results: This folder contains results and visualizations.
- requirements.txt: This file lists the Python dependencies.

## Experiments and Conclusion
We conducted extensive model training and evaluation experiments, which were tracked using the `mlflow` library. The models and their configurations were selected using the `ParameterGrid` function, which allowed us to easily explore a wide range of hyperparameters for each model.

We observed that the performance of the models varied significantly based on their configurations. Here are some of the results:

- The RandomForest model achieved an RMSE of 0.374, an MAE of 0.205, and an R2 score of 0.441 when configured with no maximum depth and 10 estimators.

- The XGBoost model achieved an RMSE of 0.319, an MAE of 0.222, and an R2 score of 0.593 when configured with a learning rate of 1.0 and 150 estimators.

More results are shown in the mlflow log.

After sorting the models based on the RMSE metric, the XGBoost model emerged as the top-performing model, with the following metrics:

- RMSE: 0.251
- MAE: 0.169
- R2: 0.749

This model was configured with a learning rate of 0.1 and 150 estimators.

![RMSE Comparison between models](https://github.com/tutalae/creditcard_approval_prediction/blob/main/results/RMSE%20scores.png)

In conclusion, the XGBoost model with a learning rate of 0.1 and 150 estimators provided the best results in terms of predicting creditworthiness. 

The success of the XGBoost model could be attributed to its gradient boosting framework, which effectively handles both linear and non-linear relationships in the data. XGBoost, an ensemble learning method, sequentially combines predictions from multiple decision trees to minimize the error rate. This gives it the ability to capture complex patterns and interactions between variables that some other models may miss.

However, we acknowledge that this conclusion is based on the specific dataset and experimental setup we used. It would be beneficial to perform additional tests with different data and configurations to validate these findings and further improve the model's performance