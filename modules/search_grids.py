from skopt.space import Real, Categorical, Integer
import numpy as np

grid_search_params = {
    "LGBMRegressor": {
        # "LGBMRegressor__n_estimators": [100, 200, 300],
        "LGBMRegressor__reg_alpha": np.linspace(10, 100, 10),
        "LGBMRegressor__reg_lambda": np.linspace(10, 100, 10),
    },
    "CatBoostRegressor": {
        "CatBoostRegressor__n_estimators": [100, 200, 300],
        "CatBoostRegressor__l2_leaf_reg": np.linspace(10, 100, 10),
    },
    "XGBRegressor": {
        "XGBRegressor__n_estimators": [100, 200, 300],
        "XGBRegressor__reg_alpha": np.linspace(10, 100, 10),
        "XGBRegressor__reg_lambda": np.linspace(10, 100, 10),
    },
}

bayes_search_params = {
    "LGBMRegressor": {
        "LGBMRegressor__n_estimators": Integer(100, 1000),
        # "LGBMRegressor__max_depth": Integer(3, 10),
        # "LGBMRegressor__learning_rate": Real(0.01, 0.3),
        # "LGBMRegressor__num_leaves": Integer(24, 80),
        # "LGBMRegressor__min_child_samples": Integer(20, 500),
        # "LGBMRegressor__subsample": Real(0.4, 1.0),
        # "LGBMRegressor__colsample_bytree": Real(0.4, 1.0),
        "LGBMRegressor__reg_alpha": Real(20, 100),
        "LGBMRegressor__reg_lambda": Real(20, 100),
    },
    "XGBRegressor": {
        "XGBRegressor__n_estimators": Integer(100, 1000),
        # "XGBRegressor__max_depth": Integer(3, 10),
        # "XGBRegressor__learning_rate": Real(0.01, 0.3),
        # "XGBRegressor__subsample": Real(0.4, 1.0),
        # "XGBRegressor__colsample_bytree": Real(0.4, 1.0),
        # "XGBRegressor__colsample_bylevel": Real(0.4, 1.0),
        # "XGBRegressor__colsample_bynode": Real(0.4, 1.0),
        "XGBRegressor__reg_alpha": Real(5, 100),
        "XGBRegressor__reg_lambda": Real(5, 100),
    },
    "CatBoostRegressor": {
        "CatBoostRegressor__n_estimators": Integer(100, 1000),
        # "CatBoostRegressor__max_depth": Integer(3, 10),
        # "CatBoostRegressor__learning_rate": Real(0.01, 0.3),
        "CatBoostRegressor__l2_leaf_reg": Real(1, 10),
    },
}
