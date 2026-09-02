import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge


X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)


Y_train_flat = Y_train.reshape(Y_train.shape[0], -1)

print("Y_train_flat shape:", Y_train_flat.shape)


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
                n_jobs=-1,
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
                n_jobs=-1,
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


results = {}

for name, model_info in models.items():

    print("\n" + "=" * 60)
    print(f"Optimizing {name}")
    print("=" * 60)

    grid_search = GridSearchCV(
        estimator=model_info["pipeline"],
        param_grid=model_info["params"],
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, Y_train_flat)

    results[name] = grid_search

    print(f"\n{name} best parameters:")
    print(grid_search.best_params_)

    print(f"\n{name} best CV MSE:")
    print(-grid_search.best_score_)


for name, grid_search in results.items():

    filename = f"{name}_model.joblib"

    joblib.dump(
        grid_search.best_estimator_,
        filename
    )

    print(f"Saved {name} model to {filename}")


joblib.dump(
    results,
    "regressor_grid_search_results.joblib"
)

print("\nSaved GridSearchCV results.")
