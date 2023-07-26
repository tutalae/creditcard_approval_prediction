# Credit Card Approval Prediction
## Overview
The goal of our project is to develop a predictive model that can accurately assess credit card applications and predict whether an applicant will be approved or rejected based on various features. By automating the credit card approval process, we aim to improve efficiency, reduce manual effort, and minimize the risk of human bias in decision-making. By better predicting defaults, the bank can reduce the default rate and better manage its credit risk and ensure its financial stability.

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
During the experimentation phase, several models including RandomForest, LogisticRegression, DecisionTree, GradientBoosting and XGBoost were evaluated using different hyperparameters. MLflow was used to track the experiments. Performance was evaluated based on RMSE, MAE, and R2 metrics.

## Running the Scripts
All the code and experiments are contained in the Jupyter notebook 'experiment_tracking_mlflow.ipynb'. You can run this notebook using Jupyter Notebook or Jupyter Lab.

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
- experiment_tracking_mlflow.ipynb: This is the main Jupyter notebook that contains all the code.
- results: This folder contains results and visualizations.
- requirements.txt: This file lists the Python dependencies.

## Experiments and Conclusion
We conducted extensive model training and evaluation experiments, which were tracked using the `mlflow` library. The models and their configurations were selected using the `ParameterGrid` function, which allowed us to easily explore a wide range of hyperparameters for each model.

The performance of the models varied significantly based on their configurations. All the results are logged and can be reviewed in the mlflow interface.

After sorting the models based on the F1 score, the Gradient Boosting model emerged as the top-performing model, with the following metrics:

- F1           0.834885
- Recall        0.80227
- Precision    0.870265
- Accuracy     0.841336

This model was configured with a learning rate of 0.1 and 50 estimators.

![F1 Scores Comparison between models](https://github.com/tutalae/creditcard_approval_prediction/blob/main/results/f1%20scores.png)

In conclusion, the Gradient Boosting model, with a learning rate of 0.1 and 50 estimators, provided the best results in terms of creditworthiness prediction.

The success of the Gradient Boosting model could be attributed to its gradient boosting framework, which effectively handles both linear and non-linear relationships in the data. Gradient Boosting is an ensemble learning method that sequentially combines predictions from multiple decision trees to minimize the error rate. This allows it to capture complex patterns and interactions between variables that some other models might miss.

However, we acknowledge that this conclusion is based on the specific dataset and experimental setup we used. It would be beneficial to perform additional tests with different data and configurations to validate these findings and further improve the model's performance.
