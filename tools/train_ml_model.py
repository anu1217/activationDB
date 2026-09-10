from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge
import joblib
import prepare_sql_adf_for_ml_model
import create_adf
import sqlite3
import argparse
import yaml

models = {


    "knn": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", KNeighborsRegressor())
        ]),

        "params": {
            "scaler": [
                StandardScaler(), 
                MinMaxScaler(),
                RobustScaler()
            ],

            "regressor__n_neighbors": [
                1,
                3,
                5,
                10,
                20
            ],

            "regressor__weights": [
                "uniform",
                "distance"
            ],

            "regressor__p": [
                1,
                2
            ]
        }
    },


    "ridge": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge())
        ]),

        "params": {
            "scaler": [
                StandardScaler(),
                MinMaxScaler(),
                RobustScaler()
            ],

            "regressor__alpha": [
                0.01,
                0.1,
                1.0,
                10.0,
                100.0
            ]
        }
    },


    "random_forest": {
        "pipeline": Pipeline([
            ("regressor", RandomForestRegressor(
                n_jobs=1,
                random_state=1
            ))
        ]),

        "params": {
            "regressor__n_estimators": [
                100,
                200
            ],

            "regressor__max_depth": [
                None,
                10,
                20
            ],

            "regressor__min_samples_leaf": [
                1,
                2,
                5
            ]
        }
    },


    "extra_trees": {
        "pipeline": Pipeline([
            ("regressor", ExtraTreesRegressor(
                n_jobs=1,
                random_state=1
            ))
        ]),

        "params": {
            "regressor__n_estimators": [
                100,
                200
            ],

            "regressor__max_depth": [
                None,
                10,
                20
            ],

            "regressor__min_samples_leaf": [
                1,
                2,
                5
            ]
        }
    }
}

def optimize_estimator_hyperparams(X_train, Y_train_flat):
    results = {}

    for regressor_name, model_info in models.items():

        grid_search = GridSearchCV(
            estimator=model_info["pipeline"],
            param_grid=model_info["params"],
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X_train, Y_train_flat)

        results[regressor_name] = grid_search
    return results

def save_optimized_model(results):
    for regressor_name, grid_search in results.items():

        fileregressor_name = f"{regressor_name}_model.joblib"

        joblib.dump(
            grid_search.best_estimator_,
            fileregressor_name
        )


    joblib.dump(
        results,
        "regressor_grid_search_results.joblib"
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_training_yaml', '-t', help="Path (str) to YAML containing information to train models")
    args = parser.parse_args()
    return args

def read_yaml(yaml_arg):
    '''
    input:
        yaml_arg : output of parse_args() corresponding to args.model_training_yaml
    '''
    with open(yaml_arg, 'r') as yaml_file:
        inputs = yaml.safe_load(yaml_file)
    return inputs


def main():
    args = parse_args()
    inputs = read_yaml(args.model_training_yaml)
    filter_str = inputs['filter_str']
    db_name = inputs['db_name']
    ordered_nucs = inputs['ordered_nucs']

    conn = sqlite3.connect(db_name)
    training_df = create_adf.make_pdf_from_sql(conn, filter_str)
    X_train, Y_train = prepare_sql_adf_for_ml_model.make_training_features_outputs(training_df, ordered_nucs)
    Y_train_flat = Y_train.reshape(Y_train.shape[0], -1)

    results = optimize_estimator_hyperparams(X_train, Y_train_flat)
    save_optimized_model(results)