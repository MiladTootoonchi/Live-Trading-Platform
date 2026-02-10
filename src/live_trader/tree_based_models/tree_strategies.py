import numpy as np

from live_trader.ml_model.utils import ProbabilisticClassifier
from live_trader.ml_model.ml_pipeline import MLStrategyBase

# Models
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


class RandomForest(MLStrategyBase):
    def __init__(self, config):
        super().__init__(config)
        self.name = "Random_Forest"
    def _initialize_model(_: np.ndarray) -> ProbabilisticClassifier:
        """
        Build and return a Random Forest classifier configured for
        probabilistic binary classification on financial time series features.

        The input array parameter is intentionally unused and is included
        for interface compatibility with model factory pipelines that
        provide feature matrices during model construction.

        Parameters
        ----------
        _ : np.ndarray
            Feature matrix placeholder (not used).

        Returns
        -------
        ProbabilisticClassifier
            A configured RandomForestClassifier instance with class balancing
            and constrained depth to reduce overfitting.
        """

        return RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )



class LGBM(MLStrategyBase):
    def __init__(self, config):
        super().__init__(config)
        self.name = "LightGBM"
    def _initialize_model(_: np.ndarray) -> ProbabilisticClassifier:
        """
        Build and return a LightGBM classifier optimized for tabular
        financial features and probabilistic prediction.

        Uses balanced class weights and moderate tree depth to handle
        noisy market data and class imbalance.

        Parameters
        ----------
        _ : np.ndarray
            Feature matrix placeholder (not used).

        Returns
        -------
        ProbabilisticClassifier
            A configured LGBMClassifier instance.
        """
        return LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=42
            )



class XGB(MLStrategyBase):
    def __init__(self, config):
        super().__init__(config)
        self.name = "XGBoost"
    def _initialize_model(_: np.ndarray) -> ProbabilisticClassifier:
        """
        Build and return an XGBoost classifier for probabilistic
        binary classification.

        Configured for stability on small-to-medium financial datasets,
        using conservative depth and subsampling to limit overfitting.

        Parameters
        ----------
        _ : np.ndarray
            Feature matrix placeholder (not used).

        Returns
        -------
        ProbabilisticClassifier
            A configured XGBClassifier instance.
        """
        return XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )



class CatBoost(MLStrategyBase):
    def __init__(self, config):
        super().__init__(config)
        self.name = "CatBoost"
    def _initialize_model(_: np.ndarray) -> ProbabilisticClassifier:
        """
        Build and return a CatBoost classifier for probabilistic
        binary classification with automatic class balancing.

        CatBoost handles feature interactions well and is robust
        to noisy datasets common in financial modeling.

        Parameters
        ----------
        _ : np.ndarray
            Feature matrix placeholder (not used).

        Returns
        -------
        ProbabilisticClassifier
            A configured CatBoostClassifier instance with silent training.
        """
        return CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                auto_class_weights="Balanced",
                random_seed=42,
                verbose=False,
            )