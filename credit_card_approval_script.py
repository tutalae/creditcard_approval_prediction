import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import xgboost as xgb
from imblearn.over_sampling import SMOTE

def load_data(path_1, path_2):
    data_1 = pd.read_parquet(path_1)
    data_2 = pd.read_parquet(path_2)

    # find all users' account open month.
    begin_month = pd.DataFrame(data_2.groupby(["ID"])["MONTHS_BALANCE"].agg(min))
    begin_month = begin_month.rename(columns={'MONTHS_BALANCE': 'begin_month'})

    data = pd.merge(data_1, begin_month, how="left", on="ID")  # merge to record data

    # Assuming 'record' is your DataFrame containing the 'STATUS' and 'dep_value' columns.
    data_2['dep_value'] = None
    data_2.loc[data_2['STATUS'].isin(['2', '3', '4', '5']), 'dep_value'] = 'Yes'

    cpunt = data_2.groupby('ID').count()
    cpunt['dep_value'][cpunt['dep_value'] > 0] = 'Yes'
    cpunt['dep_value'][cpunt['dep_value'] == 0] = 'No'
    cpunt = cpunt[['dep_value']]

    merge_data = pd.merge(data, cpunt, how='inner', on='ID')

    merge_data.dropna(inplace=True)
    merge_data.drop(['ID', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL'], axis=1, inplace=True)

    le = LabelEncoder()
    for x in merge_data:
        if merge_data[x].dtypes == 'object':
            merge_data[x] = le.fit_transform(merge_data[x])

    X = merge_data.iloc[:, 1:-1]  # X value contains all the variables except labels
    y = merge_data.iloc[:, -1]  # these are the labels

    return X, y


def generate_datasets(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    oversample = SMOTE()
    X_balanced, y_balanced = oversample.fit_resample(X_train, y_train)
    X_test_balanced, y_test_balanced = oversample.fit_resample(X_test, y_test)

    return X_balanced, X_test_balanced, y_balanced, y_test_balanced


def train_model(X_train, y_train, X_val, y_val):
    best_params = {
        'max_depth': 5,
        'min_child': 19.345653147972058,
        'objective': 'reg:linear',
        'reg_alpha': 0.031009193638004067,
        'reg_lambda': 0.013053945835415701,
        'seed': 111
    }

    train = xgb.DMatrix(X_train, label=y_train)
    validation = xgb.DMatrix(X_val, label=y_val)

    booster = xgb.train(
        params=best_params,
        dtrain=train,
        evals=[(validation, "validation")],
        num_boost_round=500,
        early_stopping_rounds=50,
    )

    return booster


def estimate_quality(model, X_val, y_val):
    validation = xgb.DMatrix(X_val, label=y_val)
    y_pred = model.predict(validation)
    return mean_squared_error(y_pred, y_val, squared=False)


if __name__ == '__main__':
    X, y = load_data('data/application_record.csv', 'data/credit_record.csv')

    print(f"data loaded")

    X_train, X_val, y_train, y_val = generate_datasets(X, y)
    print(f"datsets are generate")

    model = train_model(X_train, y_train, X_val, y_val)
    print(f"model trained")

    rmse = estimate_quality(model, X_val, y_val)
    print(f"rmse: {rmse}")
