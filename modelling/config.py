from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not installed. Run: pip install catboost")


# Model configurations with hyperparameter grids
MODEL_CONFIGS = {
    "baseline": {
        "model": DummyClassifier(strategy="most_frequent", random_state=42),
        "params": {},
    },
    
    "logistic_regression": {
        "model": LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        ),
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "saga"]
        },
    },
    
    "random_forest": {
        "model": RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        },
    },
    
    "xgboost": {
        "model": XGBClassifier(
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        ),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        },
    },
    
    "knn": {
        "model": KNeighborsClassifier(
            n_jobs=-1
        ),
        "params": {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ['uniform', 'distance'],
            "metric": ['euclidean', 'manhattan', 'minkowski'],
            "p": [1, 2]
        },
    },
    
    "gradient_boosting": {
        "model": GradientBoostingClassifier(
            random_state=42
        ),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
            "min_samples_split": [2, 5, 10]
        },
    },
    
    "hist_gradient_boosting": {
        "model": HistGradientBoostingClassifier(
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        ),
        "params": {
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [5, 10, 15, None],
            "max_iter": [100, 200, 300],
            "min_samples_leaf": [20, 50, 100],
            "l2_regularization": [0.0, 0.1, 1.0]
        },
    }
}

# Add CatBoost if available
if CATBOOST_AVAILABLE:
    MODEL_CONFIGS["catboost"] = {
        "model": CatBoostClassifier(
            random_state=42,
            verbose=0,
            auto_class_weights='Balanced',
            early_stopping_rounds=10
        ),
        "params": {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5],
            "border_count": [32, 64, 128]
        },
    }


# Single parameter configurations (no grid search - fastest training)
SINGLE_CONFIGS = {
    "baseline": MODEL_CONFIGS["baseline"],
    
    "logistic_regression": {
        **MODEL_CONFIGS["logistic_regression"],
        "params": {
            "C": [1.0],
            "solver": ["lbfgs"]
        }
    },
    
    "random_forest": {
        **MODEL_CONFIGS["random_forest"],
        "params": {
            "n_estimators": [200],
            "max_depth": [20],
            "min_samples_split": [5],
            "min_samples_leaf": [2],
            "max_features": ["sqrt"]
        }
    },
    
    "xgboost": {
        **MODEL_CONFIGS["xgboost"],
        "params": {
            "n_estimators": [250],
            "max_depth": [7],
            "learning_rate": [0.05],
            "min_child_weight": [2],
            "subsample": [0.9],
            "colsample_bytree": [0.9],
        }
    },
    
    "knn": {
        **MODEL_CONFIGS["knn"],
        "params": {
            "n_neighbors": [7],
            "weights": ["distance"],
            "metric": ["euclidean"],
            "p": [2]
        }
    },
    
    "gradient_boosting": {
        **MODEL_CONFIGS["gradient_boosting"],
        "params": {
            "n_estimators": [200],
            "max_depth": [5],
            "learning_rate": [0.05],
            "subsample": [0.9],
            "min_samples_split": [5]
        }
    },
    
    "hist_gradient_boosting": {
        **MODEL_CONFIGS["hist_gradient_boosting"],
        "params": {
            "learning_rate": [0.1],
            "max_depth": [10],
            "max_iter": [250],
            "min_samples_leaf": [50],
            "l2_regularization": [0.1]
        }
    }
}

# Add CatBoost single config if available
if CATBOOST_AVAILABLE:
    SINGLE_CONFIGS["catboost"] = {
        **MODEL_CONFIGS["catboost"],
        "params": {
            "iterations": [250],
            "depth": [7],
            "learning_rate": [0.07],
            "l2_leaf_reg": [3],
            "border_count": [64]
        }
    }


# Categorical features that need encoding
CATEGORICAL_FEATURES = [
    'property_type',
    'room_type', 
    'city',
    'neighbourhood',
]

# Numeric features (no encoding needed)
NUMERIC_FEATURES = [
    'accommodates',
    'bathrooms',
    'bedrooms',
    'host_response_rate',
    'amenity_count',
    'price_per_bed',
    'listing_density',
    'price_relative_to_room_type',
    'log_price',
    'log_number_of_reviews'
]

# Features to exclude from X (targets)
TARGET_FEATURES = ['rating_category']

TRAKING_URI = "http://127.0.0.1:5000"