import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def _haversine(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


class DemandPredictor:

    def __init__(self, output_dir="models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_names = None
        self.metrics = {}

    def prepare_features(self, bus_stops, metro_stations, population_grid):
        """Build feature matrix for demand prediction."""
        station_lats = metro_stations["latitude"].values
        station_lons = metro_stations["longitude"].values

        features = []
        for _, stop in bus_stops.iterrows():
            distances = np.array([
                _haversine(stop["latitude"], stop["longitude"], slat, slon)
                for slat, slon in zip(station_lats, station_lons)
            ])
            nearest_dist = distances.min()
            n_stations_1km = int(np.sum(distances < 1.0))
            n_stations_2km = int(np.sum(distances < 2.0))

            # Find nearby grid cells for aggregated features
            nearby_cells = population_grid[
                (abs(population_grid["latitude"] - stop["latitude"]) < 0.02) &
                (abs(population_grid["longitude"] - stop["longitude"]) < 0.02)
            ]

            if len(nearby_cells) > 0:
                avg_density = nearby_cells["population_density"].mean()
                max_density = nearby_cells["population_density"].max()
                avg_commercial = nearby_cells["commercial_activity"].mean()
                avg_walkability = nearby_cells["walkability_score"].mean()
                avg_income = nearby_cells["avg_income"].mean()
                residential_ratio = (nearby_cells["commercial_activity"] < 0.4).mean()
            else:
                avg_density = stop["surrounding_density"]
                max_density = stop["surrounding_density"]
                avg_commercial = stop["commercial_activity"]
                avg_walkability = stop["walkability_score"]
                avg_income = 8000
                residential_ratio = 0.5

            features.append({
                "stop_id": stop["stop_id"],
                "latitude": stop["latitude"],
                "longitude": stop["longitude"],
                "dist_nearest_metro": round(nearest_dist, 3),
                "n_stations_1km": n_stations_1km,
                "n_stations_2km": n_stations_2km,
                "avg_density": round(avg_density, 1),
                "max_density": round(max_density, 1),
                "commercial_activity": round(avg_commercial, 3),
                "walkability_score": round(avg_walkability, 3),
                "avg_income": round(avg_income, 0),
                "residential_ratio": round(residential_ratio, 3),
            })

        df = pd.DataFrame(features)
        print(f"Prepared {len(df)} feature vectors for demand prediction")
        return df

    def generate_demand_labels(self, feature_df):
        """Generate synthetic boarding demand labels based on features.

        Args:
            feature_df: Feature DataFrame from prepare_features().

        Returns:
            Series of demand values.
        """
        rng = np.random.default_rng(42)

        demand = (
            feature_df["avg_density"] * 0.15
            + feature_df["commercial_activity"] * 3000
            + feature_df["walkability_score"] * 1500
            - feature_df["dist_nearest_metro"] * 200
            + feature_df["n_stations_2km"] * 100
            + feature_df["avg_income"] * 0.02
        )
        demand = np.maximum(demand, 50)
        demand += rng.normal(0, demand * 0.1)
        demand = np.maximum(demand, 10)

        return pd.Series(np.round(demand, 1), name="boarding_demand")

    def train(self, feature_df, demand):
        """Train the XGBoost demand prediction model."""
        self.feature_names = [
            "dist_nearest_metro", "n_stations_1km", "n_stations_2km",
            "avg_density", "max_density", "commercial_activity",
            "walkability_score", "avg_income", "residential_ratio",
        ]
        X = feature_df[self.feature_names].values
        y = demand.values

        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        self.model.fit(X, y)
        print(f"Trained XGBoost demand model on {len(X)} samples")

        return self.model

    def spatial_cross_validate(self, feature_df, demand, n_folds=5):
        """Run spatial cross-validation using geographic folds.

        Splits data into spatial blocks based on latitude to avoid
        spatial autocorrelation leakage between train and test sets.

        Args:
            feature_df: Feature DataFrame.
            demand: Target demand values.
            n_folds: Number of spatial folds.

        Returns:
            Dict with cross-validation metrics.
        """
        X = feature_df[self.feature_names].values
        y = demand.values

        # Sort by latitude for spatial splits
        lat_order = np.argsort(feature_df["latitude"].values)
        fold_indices = np.array_split(lat_order, n_folds)

        mae_scores = []
        rmse_scores = []
        r2_scores = []

        for fold_idx in range(n_folds):
            test_idx = fold_indices[fold_idx]
            train_idx = np.concatenate([fold_indices[i] for i in range(n_folds) if i != fold_idx])

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae_scores.append(mean_absolute_error(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2_scores.append(r2_score(y_test, y_pred))

        self.metrics = {
            "mae_mean": round(np.mean(mae_scores), 2),
            "mae_std": round(np.std(mae_scores), 2),
            "rmse_mean": round(np.mean(rmse_scores), 2),
            "rmse_std": round(np.std(rmse_scores), 2),
            "r2_mean": round(np.mean(r2_scores), 4),
            "r2_std": round(np.std(r2_scores), 4),
        }

        print(f"Spatial CV ({n_folds}-fold): MAE={self.metrics['mae_mean']:.2f} "
              f"(+/-{self.metrics['mae_std']:.2f}), R2={self.metrics['r2_mean']:.4f}")

        return self.metrics

    def predict(self, feature_df):
        """Predict boarding demand for bus stops."""
        X = feature_df[self.feature_names].values
        return self.model.predict(X)

    def get_feature_importance(self):
        """Return feature importance from the trained model.

        Returns:
            DataFrame with feature names and importance scores.
        """
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return df

    def generate_demand_heatmap(self, feature_df, predictions):
        """Create a heatmap-ready DataFrame of predicted demand.

        Args:
            feature_df: Feature DataFrame with coordinates.
            predictions: Predicted demand values.

        Returns:
            DataFrame with latitude, longitude, and demand columns.
        """
        heatmap_data = pd.DataFrame({
            "latitude": feature_df["latitude"],
            "longitude": feature_df["longitude"],
            "demand": predictions,
        })
        heatmap_data = heatmap_data.sort_values("demand", ascending=False)
        print(f"Generated demand heatmap: {len(heatmap_data)} points")
        return heatmap_data

    def save_model(self, name="demand_model"):
        """Save the trained model to disk.

        Args:
            name: Filename without extension.
        """
        path = self.output_dir / f"{name}.pkl"
        joblib.dump({"model": self.model, "features": self.feature_names}, path)
        print(f"Demand model saved to {path}")

    def load_model(self, name="demand_model"):
        """Load a trained model from disk.

        Args:
            name: Filename without extension.
        """
        path = self.output_dir / f"{name}.pkl"
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["features"]
        print(f"Demand model loaded from {path}")
