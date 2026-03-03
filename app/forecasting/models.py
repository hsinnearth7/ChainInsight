"""Forecasting model zoo with unified fit/predict interface (Strategy pattern).

All models implement the ForecastModel protocol:
    - fit(Y_train, X_train=None) → self
    - predict(h, X_future=None) → pd.DataFrame with (unique_id, ds, y_hat)
    - name → str

Models:
    1. NaiveMovingAverage  — rolling mean baseline
    2. SARIMAXForecaster   — seasonal ARIMA with exogenous
    3. XGBoostForecaster   — gradient boosting with lag features
    4. LightGBMForecaster  — LightGBM with lag features
    5. ChronosForecaster   — Amazon Chronos-2 zero-shot (foundation model)
    6. RoutingEnsemble     — cold-start routing logic
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from app.logging import get_logger
from app.settings import get_model_config

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------


class ForecastModel(ABC):
    """Unified forecast model interface (Strategy pattern)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier."""

    @abstractmethod
    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "ForecastModel":
        """Fit model on training data.

        Args:
            Y_train: Nixtla format (unique_id, ds, y).
            X_train: Optional exogenous features (unique_id, ds, ...).

        Returns:
            self for chaining.
        """

    @abstractmethod
    def predict(
        self,
        h: int,
        X_future: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts.

        Args:
            h: Forecast horizon (number of periods ahead).
            X_future: Optional future exogenous features.

        Returns:
            DataFrame with columns (unique_id, ds, y_hat).
        """


# ---------------------------------------------------------------------------
# 1. Naive Moving Average
# ---------------------------------------------------------------------------


class NaiveMovingAverage(ForecastModel):
    """Simple moving average baseline."""

    def __init__(self, window: int = 30):
        self.window = window
        self._last_values: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return f"naive_ma{self.window}"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "NaiveMovingAverage":
        for uid, grp in Y_train.groupby("unique_id"):
            values = grp.sort_values("ds")["y"].values
            self._last_values[str(uid)] = values[-self.window :]
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        records = []
        for uid, values in self._last_values.items():
            ma = float(np.mean(values))
            last_ds = pd.Timestamp("2024-01-01")  # will be overridden by caller
            for step in range(h):
                records.append({"unique_id": uid, "ds": last_ds + pd.Timedelta(days=step + 1), "y_hat": max(0, ma)})
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. SARIMAX
# ---------------------------------------------------------------------------


class SARIMAXForecaster(ForecastModel):
    """Seasonal ARIMA with exogenous variables.

    Uses statsforecast for fast fitting when available, falls back to statsmodels.
    Good for: cold-start SKUs, seasonal patterns, intermittent demand.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 7),
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self._models: dict[str, Any] = {}
        self._last_dates: dict[str, pd.Timestamp] = {}

    @property
    def name(self) -> str:
        return "sarimax"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "SARIMAXForecaster":
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA

            sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D", n_jobs=1)
            sf.fit(Y_train)
            self._sf = sf
            self._use_statsforecast = True
        except ImportError:
            self._use_statsforecast = False
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            for uid, grp in Y_train.groupby("unique_id"):
                grp = grp.sort_values("ds")
                y = grp["y"].values
                if len(y) < 14:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                        fitted = model.fit(disp=False, maxiter=50)
                        self._models[str(uid)] = fitted
                        self._last_dates[str(uid)] = grp["ds"].iloc[-1]
                except Exception:
                    logger.warning("sarimax_fit_failed", uid=uid)
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if getattr(self, "_use_statsforecast", False):
            forecasts = self._sf.predict(h=h)
            forecasts = forecasts.reset_index()
            forecasts = forecasts.rename(columns={"AutoARIMA": "y_hat"})
            forecasts["y_hat"] = forecasts["y_hat"].clip(lower=0)
            return forecasts[["unique_id", "ds", "y_hat"]]

        records = []
        for uid, model in self._models.items():
            try:
                forecast = model.forecast(steps=h)
                last_ds = self._last_dates[uid]
                future_dates = pd.date_range(start=last_ds + pd.Timedelta(days=1), periods=h, freq="D")
                for ds, yhat in zip(future_dates, forecast):
                    records.append({"unique_id": uid, "ds": ds, "y_hat": max(0, float(yhat))})
            except Exception:
                logger.warning("sarimax_predict_failed", uid=uid)
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. XGBoost Forecaster
# ---------------------------------------------------------------------------


def _build_lag_features(Y_df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Build lag and rolling features for tree-based models."""
    if lags is None:
        lags = [1, 7, 14, 28]

    frames = []
    for uid, grp in Y_df.groupby("unique_id"):
        grp = grp.sort_values("ds").copy()
        for lag in lags:
            grp[f"lag_{lag}"] = grp["y"].shift(lag)
        grp["rolling_mean_7"] = grp["y"].shift(1).rolling(7, min_periods=1).mean()
        grp["rolling_std_7"] = grp["y"].shift(1).rolling(7, min_periods=1).std().fillna(0)
        grp["rolling_mean_28"] = grp["y"].shift(1).rolling(28, min_periods=1).mean()
        grp["day_of_week"] = grp["ds"].dt.dayofweek
        grp["month"] = grp["ds"].dt.month
        grp["day_of_year"] = grp["ds"].dt.dayofyear
        frames.append(grp)

    result = pd.concat(frames, ignore_index=True)
    return result


FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_28",
    "day_of_week", "month", "day_of_year",
]


class XGBoostForecaster(ForecastModel):
    """XGBoost time series forecaster with lag features.

    Good for: feature interactions, non-linear patterns, mature SKUs.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 5, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model = None
        self._train_data: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "XGBoostForecaster":
        from xgboost import XGBRegressor

        featured = _build_lag_features(Y_train)
        featured = featured.dropna(subset=FEATURE_COLS)

        X = featured[FEATURE_COLS].values
        y = featured["y"].values

        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            n_jobs=1,
        )
        self._model.fit(X, y)
        self._train_data = Y_train.copy()
        logger.info("model_fitted", model="xgboost", n_samples=len(X))
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._model is None or self._train_data is None:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat"])

        records = []
        for uid, grp in self._train_data.groupby("unique_id"):
            grp = grp.sort_values("ds")
            history = grp["y"].values.tolist()
            last_ds = grp["ds"].iloc[-1]

            for step in range(h):
                ds = last_ds + pd.Timedelta(days=step + 1)
                features = self._compute_features(history, ds)
                yhat = float(self._model.predict(np.array([features]))[0])
                yhat = max(0, yhat)
                records.append({"unique_id": uid, "ds": ds, "y_hat": yhat})
                history.append(yhat)

        return pd.DataFrame(records)

    def _compute_features(self, history: list[float], ds: pd.Timestamp) -> list[float]:
        """Compute lag features for a single prediction step."""
        n = len(history)
        lag_1 = history[-1] if n >= 1 else 0
        lag_7 = history[-7] if n >= 7 else lag_1
        lag_14 = history[-14] if n >= 14 else lag_1
        lag_28 = history[-28] if n >= 28 else lag_1
        rm7 = float(np.mean(history[-7:])) if n >= 7 else float(np.mean(history))
        rs7 = float(np.std(history[-7:])) if n >= 7 else 0
        rm28 = float(np.mean(history[-28:])) if n >= 28 else float(np.mean(history))
        return [lag_1, lag_7, lag_14, lag_28, rm7, rs7, rm28, ds.dayofweek, ds.month, ds.dayofyear]

    @property
    def feature_importance(self) -> dict[str, float] | None:
        """Return feature importances if model is fitted."""
        if self._model is None:
            return None
        return dict(zip(FEATURE_COLS, self._model.feature_importances_))


# ---------------------------------------------------------------------------
# 4. LightGBM Forecaster
# ---------------------------------------------------------------------------


class LightGBMForecaster(ForecastModel):
    """LightGBM time series forecaster with lag features.

    Best overall MAPE for mature SKUs. Fast training and inference.
    """

    def __init__(self, n_estimators: int = 300, num_leaves: int = 31, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self._model = None
        self._train_data: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "lightgbm"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "LightGBMForecaster":
        import lightgbm as lgb

        featured = _build_lag_features(Y_train)
        featured = featured.dropna(subset=FEATURE_COLS)

        X = featured[FEATURE_COLS].values
        y = featured["y"].values

        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        self._model.fit(X, y)
        self._train_data = Y_train.copy()
        logger.info("model_fitted", model="lightgbm", n_samples=len(X))
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._model is None or self._train_data is None:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat"])

        records = []
        for uid, grp in self._train_data.groupby("unique_id"):
            grp = grp.sort_values("ds")
            history = grp["y"].values.tolist()
            last_ds = grp["ds"].iloc[-1]

            for step in range(h):
                ds = last_ds + pd.Timedelta(days=step + 1)
                features = XGBoostForecaster._compute_features(None, history, ds)
                yhat = float(self._model.predict(np.array([features]))[0])
                yhat = max(0, yhat)
                records.append({"unique_id": uid, "ds": ds, "y_hat": yhat})
                history.append(yhat)

        return pd.DataFrame(records)

    @property
    def feature_importance(self) -> dict[str, float] | None:
        if self._model is None:
            return None
        return dict(zip(FEATURE_COLS, self._model.feature_importances_))


# ---------------------------------------------------------------------------
# 5. Chronos-2 Zero-Shot Forecaster
# ---------------------------------------------------------------------------


class ChronosForecaster(ForecastModel):
    """Amazon Chronos-2 foundation model for zero-shot forecasting.

    No training needed — uses pretrained time series knowledge.
    Good for: cold-start SKUs, benchmark baseline.
    """

    def __init__(self, model_name: str = "amazon/chronos-t5-small", prediction_length: int = 14):
        self.model_name = model_name
        self.prediction_length = prediction_length
        self._pipeline = None
        self._train_data: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "chronos2_zs"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "ChronosForecaster":
        """Chronos is zero-shot — fit just stores data for predict."""
        self._train_data = Y_train.copy()
        try:
            import torch
            from chronos import ChronosPipeline

            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            logger.info("chronos_loaded", model=self.model_name)
        except ImportError:
            logger.warning("chronos_not_installed", fallback="naive_ma30")
            self._pipeline = None
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._train_data is None:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat"])

        records = []
        h = min(h, self.prediction_length)

        for uid, grp in self._train_data.groupby("unique_id"):
            grp = grp.sort_values("ds")
            last_ds = grp["ds"].iloc[-1]
            future_dates = pd.date_range(start=last_ds + pd.Timedelta(days=1), periods=h, freq="D")

            if self._pipeline is not None:
                import torch

                context = torch.tensor(grp["y"].values[-512:], dtype=torch.float32).unsqueeze(0)
                forecast = self._pipeline.predict(context, prediction_length=h)
                median = forecast.median(dim=1).values.squeeze().numpy()
                for ds, yhat in zip(future_dates, median):
                    records.append({"unique_id": uid, "ds": ds, "y_hat": max(0, float(yhat))})
            else:
                # Fallback: simple moving average
                ma = float(grp["y"].tail(30).mean())
                for ds in future_dates:
                    records.append({"unique_id": uid, "ds": ds, "y_hat": max(0, ma)})

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. Routing Ensemble
# ---------------------------------------------------------------------------


class RoutingEnsemble(ForecastModel):
    """Intelligent routing ensemble based on SKU characteristics.

    Routing logic:
        - history < threshold_days → Chronos-2 zero-shot (cold start)
        - intermittency > 0.5      → SARIMAX (handles zeros well)
        - otherwise                → LightGBM (best overall MAPE)
    """

    def __init__(
        self,
        cold_start_threshold_days: int = 60,
        intermittency_threshold: float = 0.5,
    ):
        self.cold_start_threshold_days = cold_start_threshold_days
        self.intermittency_threshold = intermittency_threshold
        self._models: dict[str, ForecastModel] = {}
        self._routing_decisions: dict[str, str] = {}
        self._sku_stats: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "routing_ensemble"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "RoutingEnsemble":
        # Compute per-SKU statistics for routing
        for uid, grp in Y_train.groupby("unique_id"):
            n_days = len(grp)
            intermittency = float((grp["y"] == 0).mean())
            self._sku_stats[str(uid)] = {
                "n_days": n_days,
                "intermittency": intermittency,
            }

            # Routing decision
            if n_days < self.cold_start_threshold_days:
                self._routing_decisions[str(uid)] = "chronos2_zs"
            elif intermittency > self.intermittency_threshold:
                self._routing_decisions[str(uid)] = "sarimax"
            else:
                self._routing_decisions[str(uid)] = "lightgbm"

        # Determine which models are needed
        needed_models = set(self._routing_decisions.values())
        logger.info(
            "routing_decisions",
            total_skus=len(self._routing_decisions),
            cold_start=sum(1 for v in self._routing_decisions.values() if v == "chronos2_zs"),
            intermittent=sum(1 for v in self._routing_decisions.values() if v == "sarimax"),
            mature=sum(1 for v in self._routing_decisions.values() if v == "lightgbm"),
        )

        # Fit each needed model on its assigned SKUs
        model_config = get_model_config()

        for model_name in needed_models:
            assigned_uids = {uid for uid, m in self._routing_decisions.items() if m == model_name}
            subset = Y_train[Y_train["unique_id"].isin(assigned_uids)]

            if model_name == "chronos2_zs":
                cfg = model_config.get("chronos", {})
                model = ChronosForecaster(
                    model_name=cfg.get("model_name", "amazon/chronos-t5-small"),
                    prediction_length=cfg.get("prediction_length", 14),
                )
            elif model_name == "sarimax":
                cfg = model_config.get("sarimax", {})
                model = SARIMAXForecaster(
                    order=tuple(cfg.get("order", [1, 1, 1])),
                    seasonal_order=tuple(cfg.get("seasonal_order", [1, 1, 1, 7])),
                )
            else:  # lightgbm
                cfg = model_config.get("lightgbm", {})
                model = LightGBMForecaster(
                    n_estimators=cfg.get("n_estimators", 300),
                    num_leaves=cfg.get("num_leaves", 31),
                    learning_rate=cfg.get("learning_rate", 0.1),
                )

            model.fit(subset)
            self._models[model_name] = model

        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        all_forecasts = []
        for model_name, model in self._models.items():
            forecasts = model.predict(h=h, X_future=X_future)
            forecasts["model"] = model_name
            all_forecasts.append(forecasts)

        if not all_forecasts:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat", "model"])

        return pd.concat(all_forecasts, ignore_index=True)

    @property
    def routing_summary(self) -> dict[str, int]:
        """Summary of routing decisions."""
        summary: dict[str, int] = {}
        for model_name in self._routing_decisions.values():
            summary[model_name] = summary.get(model_name, 0) + 1
        return summary


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class ForecastModelFactory:
    """Factory for creating forecast models from configuration."""

    _registry: dict[str, type[ForecastModel]] = {
        "naive_ma30": NaiveMovingAverage,
        "sarimax": SARIMAXForecaster,
        "xgboost": XGBoostForecaster,
        "lightgbm": LightGBMForecaster,
        "chronos2_zs": ChronosForecaster,
        "routing_ensemble": RoutingEnsemble,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> ForecastModel:
        """Create a forecast model by name.

        Args:
            name: Model name key.
            **kwargs: Model-specific parameters.

        Returns:
            Initialized ForecastModel instance.
        """
        model_cls = cls._registry.get(name)
        if model_cls is None:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._registry.keys())}")
        return model_cls(**kwargs)

    @classmethod
    def create_all(cls) -> dict[str, ForecastModel]:
        """Create all registered models with default configs."""
        config = get_model_config()
        models = {}

        models["naive_ma30"] = NaiveMovingAverage(window=config.get("naive", {}).get("window", 30))

        sarimax_cfg = config.get("sarimax", {})
        models["sarimax"] = SARIMAXForecaster(
            order=tuple(sarimax_cfg.get("order", [1, 1, 1])),
            seasonal_order=tuple(sarimax_cfg.get("seasonal_order", [1, 1, 1, 7])),
        )

        xgb_cfg = config.get("xgboost", {})
        models["xgboost"] = XGBoostForecaster(
            n_estimators=xgb_cfg.get("n_estimators", 200),
            max_depth=xgb_cfg.get("max_depth", 5),
            learning_rate=xgb_cfg.get("learning_rate", 0.1),
        )

        lgb_cfg = config.get("lightgbm", {})
        models["lightgbm"] = LightGBMForecaster(
            n_estimators=lgb_cfg.get("n_estimators", 300),
            num_leaves=lgb_cfg.get("num_leaves", 31),
            learning_rate=lgb_cfg.get("learning_rate", 0.1),
        )

        chronos_cfg = config.get("chronos", {})
        models["chronos2_zs"] = ChronosForecaster(
            model_name=chronos_cfg.get("model_name", "amazon/chronos-t5-small"),
            prediction_length=chronos_cfg.get("prediction_length", 14),
        )

        routing_cfg = config.get("routing", {})
        models["routing_ensemble"] = RoutingEnsemble(
            cold_start_threshold_days=routing_cfg.get("cold_start_threshold_days", 60),
            intermittency_threshold=routing_cfg.get("intermittency_threshold", 0.5),
        )

        return models

    @classmethod
    def available_models(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._registry.keys())
