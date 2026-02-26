import numpy as np
import pandas as pd
from pathlib import Path


# Riyadh center coordinates
RIYADH_LAT = 24.7136
RIYADH_LON = 46.6753
GRID_SIZE = 30  # km
CELL_SIZE = 1   # km

# District centers (approximate lat/lon)
DISTRICTS = {
    "Al Olaya": {"lat": 24.6900, "lon": 46.6850, "type": "commercial", "density": 12000},
    "Al Naseem": {"lat": 24.7300, "lon": 46.7400, "type": "residential", "density": 9500},
    "KAFD": {"lat": 24.7670, "lon": 46.6380, "type": "business", "density": 8000},
    "DQ": {"lat": 24.6700, "lon": 46.6250, "type": "diplomatic", "density": 3500},
    "Airport": {"lat": 24.9570, "lon": 46.6990, "type": "transport", "density": 2000},
    "Al Malaz": {"lat": 24.6680, "lon": 46.7260, "type": "residential", "density": 11000},
    "Al Murabba": {"lat": 24.6510, "lon": 46.7110, "type": "mixed", "density": 7500},
    "Al Shifa": {"lat": 24.5900, "lon": 46.6900, "type": "residential", "density": 8500},
    "Irqah": {"lat": 24.7100, "lon": 46.5700, "type": "residential", "density": 5000},
    "Al Sulay": {"lat": 24.6100, "lon": 46.7700, "type": "residential", "density": 7000},
}

# Metro lines with approximate station coordinates
METRO_LINES = {
    "Line 1 (Blue)": {
        "color": "#0000FF",
        "stations": [
            (24.9100, 46.6950), (24.8900, 46.6920), (24.8700, 46.6880),
            (24.8500, 46.6850), (24.8300, 46.6820), (24.8100, 46.6800),
            (24.7900, 46.6770), (24.7700, 46.6740), (24.7500, 46.6710),
            (24.7300, 46.6680), (24.7100, 46.6650), (24.6900, 46.6620),
            (24.6700, 46.6600), (24.6500, 46.6580), (24.6300, 46.6560),
            (24.6100, 46.6540),
        ],
    },
    "Line 2 (Green)": {
        "color": "#00AA00",
        "stations": [
            (24.7200, 46.5800), (24.7180, 46.6000), (24.7160, 46.6200),
            (24.7140, 46.6400), (24.7120, 46.6600), (24.7100, 46.6800),
            (24.7080, 46.7000), (24.7060, 46.7200), (24.7040, 46.7400),
            (24.7020, 46.7600), (24.7000, 46.7800), (24.6980, 46.8000),
            (24.6960, 46.8200),
        ],
    },
    "Line 3 (Orange)": {
        "color": "#FF8800",
        "stations": [
            (24.7700, 46.6100), (24.7650, 46.6250), (24.7600, 46.6400),
            (24.7550, 46.6550), (24.7500, 46.6700), (24.7450, 46.6850),
            (24.7400, 46.7000), (24.7350, 46.7150), (24.7300, 46.7300),
            (24.7250, 46.7450), (24.7200, 46.7600), (24.7150, 46.7750),
            (24.7100, 46.7900), (24.7050, 46.8050),
        ],
    },
    "Line 4 (Yellow)": {
        "color": "#FFD700",
        "stations": [
            (24.7800, 46.6380), (24.7700, 46.6420), (24.7600, 46.6460),
            (24.7500, 46.6500), (24.7400, 46.6540), (24.7300, 46.6580),
            (24.7200, 46.6620), (24.7100, 46.6660), (24.7000, 46.6700),
            (24.6900, 46.6740), (24.6800, 46.6780), (24.6700, 46.6820),
            (24.6600, 46.6860), (24.6500, 46.6900),
        ],
    },
    "Line 5 (Purple)": {
        "color": "#9900CC",
        "stations": [
            (24.7500, 46.6200), (24.7400, 46.6300), (24.7300, 46.6450),
            (24.7200, 46.6600), (24.7100, 46.6750), (24.7000, 46.6900),
            (24.6900, 46.7050), (24.6800, 46.7200), (24.6700, 46.7350),
            (24.6600, 46.7500), (24.6500, 46.7650), (24.6400, 46.7800),
            (24.6300, 46.7950), (24.6200, 46.8100),
        ],
    },
    "Line 6 (Red)": {
        "color": "#FF0000",
        "stations": [
            (24.6300, 46.6200), (24.6400, 46.6350), (24.6500, 46.6500),
            (24.6600, 46.6650), (24.6700, 46.6800), (24.6800, 46.6950),
            (24.6900, 46.7100), (24.7000, 46.7250), (24.7100, 46.7400),
            (24.7200, 46.7550), (24.7300, 46.7700), (24.7400, 46.7850),
            (24.7500, 46.8000), (24.7600, 46.8150),
        ],
    },
}


def _km_to_deg_lat(km):
    """Convert kilometers to degrees latitude."""
    return km / 111.0


def _km_to_deg_lon(km, lat):
    """Convert kilometers to degrees longitude at a given latitude."""
    return km / (111.0 * np.cos(np.radians(lat)))


def _haversine(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points in km.

    Args:
        lat1: Latitude of point 1.
        lon1: Longitude of point 1.
        lat2: Latitude of point 2.
        lon2: Longitude of point 2.

    Returns:
        Distance in kilometers.
    """
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


class RiyadhDataGenerator:

    def __init__(self, seed=42, output_dir="data"):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_population_grid(self):
        """Generate a 30x30 km population density grid centered on Riyadh.

        Each cell contains population density, commercial activity,
        walkability score, and average income attributes.

        Returns:
            DataFrame with one row per grid cell.
        """
        half = GRID_SIZE / 2
        lat_min = RIYADH_LAT - _km_to_deg_lat(half)
        lon_min = RIYADH_LON - _km_to_deg_lon(half, RIYADH_LAT)

        rows = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell_lat = lat_min + _km_to_deg_lat(i + 0.5)
                cell_lon = lon_min + _km_to_deg_lon(j + 0.5, RIYADH_LAT)

                # Base density falls off from center
                dist_center = _haversine(cell_lat, cell_lon, RIYADH_LAT, RIYADH_LON)
                base_density = max(500, 8000 * np.exp(-0.08 * dist_center))

                # Add district influence
                commercial = 0.2
                walkability = 0.4
                avg_income = 8000

                for name, info in DISTRICTS.items():
                    dist_d = _haversine(cell_lat, cell_lon, info["lat"], info["lon"])
                    influence = np.exp(-0.5 * dist_d)
                    base_density += info["density"] * influence * 0.3

                    if info["type"] == "commercial":
                        commercial += 0.6 * influence
                    elif info["type"] == "business":
                        commercial += 0.5 * influence
                        avg_income += 12000 * influence

                    if info["type"] in ("commercial", "mixed"):
                        walkability += 0.4 * influence

                    if info["type"] == "diplomatic":
                        avg_income += 15000 * influence

                # Add noise
                density = max(100, base_density + self.rng.normal(0, 500))
                commercial = np.clip(commercial + self.rng.normal(0, 0.05), 0, 1)
                walkability = np.clip(walkability + self.rng.normal(0, 0.05), 0, 1)
                avg_income = max(3000, avg_income + self.rng.normal(0, 1000))

                rows.append({
                    "grid_i": i,
                    "grid_j": j,
                    "latitude": round(cell_lat, 6),
                    "longitude": round(cell_lon, 6),
                    "population_density": round(density, 1),
                    "commercial_activity": round(commercial, 3),
                    "walkability_score": round(walkability, 3),
                    "avg_income": round(avg_income, 0),
                })

        df = pd.DataFrame(rows)
        print(f"Generated population grid: {len(df)} cells ({GRID_SIZE}x{GRID_SIZE} km)")
        return df

    def generate_metro_stations(self):
        """Generate metro station locations across 6 lines.

        Returns:
            DataFrame with station coordinates and line assignments.
        """
        rows = []
        station_id = 0
        for line_name, line_info in METRO_LINES.items():
            for idx, (lat, lon) in enumerate(line_info["stations"]):
                rows.append({
                    "station_id": station_id,
                    "station_name": f"{line_name.split('(')[1].rstrip(')')} S{idx + 1}",
                    "line": line_name,
                    "line_color": line_info["color"],
                    "latitude": lat,
                    "longitude": lon,
                })
                station_id += 1

        df = pd.DataFrame(rows)
        print(f"Generated {len(df)} metro stations across {len(METRO_LINES)} lines")
        return df

    def generate_candidate_bus_stops(self, population_grid, n_stops=500):
        """Generate candidate bus stop locations weighted by population density.

        Args:
            population_grid: Population density grid DataFrame.
            n_stops: Number of candidate stops to generate.

        Returns:
            DataFrame with bus stop coordinates and attributes.
        """
        weights = population_grid["population_density"].values
        weights = weights / weights.sum()
        chosen = self.rng.choice(len(population_grid), size=n_stops, p=weights, replace=True)

        rows = []
        for stop_id, cell_idx in enumerate(chosen):
            cell = population_grid.iloc[cell_idx]
            offset_lat = self.rng.uniform(-0.004, 0.004)
            offset_lon = self.rng.uniform(-0.004, 0.004)

            rows.append({
                "stop_id": stop_id,
                "latitude": round(cell["latitude"] + offset_lat, 6),
                "longitude": round(cell["longitude"] + offset_lon, 6),
                "surrounding_density": cell["population_density"],
                "commercial_activity": cell["commercial_activity"],
                "walkability_score": cell["walkability_score"],
            })

        df = pd.DataFrame(rows)
        print(f"Generated {len(df)} candidate bus stops")
        return df

    def generate_od_demand(self, population_grid, metro_stations):
        """Generate origin-destination demand matrix from cells to metro stations.

        Args:
            population_grid: Population grid DataFrame.
            metro_stations: Metro stations DataFrame.

        Returns:
            DataFrame with origin cell, destination station, and demand.
        """
        station_lats = metro_stations["latitude"].values
        station_lons = metro_stations["longitude"].values

        rows = []
        for _, cell in population_grid.iterrows():
            distances = np.array([
                _haversine(cell["latitude"], cell["longitude"], slat, slon)
                for slat, slon in zip(station_lats, station_lons)
            ])

            nearest_idx = np.argmin(distances)
            nearest_dist = distances[nearest_idx]

            # Demand model: higher density + shorter distance = more demand
            demand = cell["population_density"] * np.exp(-0.3 * nearest_dist)
            demand *= (1 + 0.5 * cell["commercial_activity"])
            demand = max(0, demand + self.rng.normal(0, demand * 0.1))

            rows.append({
                "origin_lat": cell["latitude"],
                "origin_lon": cell["longitude"],
                "dest_station_id": int(metro_stations.iloc[nearest_idx]["station_id"]),
                "distance_km": round(nearest_dist, 2),
                "demand": round(demand, 1),
            })

        df = pd.DataFrame(rows)
        print(f"Generated OD demand matrix: {len(df)} cell-station pairs")
        return df

    def generate_all(self):
        """Generate all datasets and save to disk.

        Returns:
            Dict mapping dataset names to DataFrames.
        """
        print("Generating Riyadh transit network data...")
        print("-" * 50)

        grid = self.generate_population_grid()
        stations = self.generate_metro_stations()
        bus_stops = self.generate_candidate_bus_stops(grid)
        od_demand = self.generate_od_demand(grid, stations)

        grid.to_csv(self.output_dir / "population_grid.csv", index=False)
        stations.to_csv(self.output_dir / "metro_stations.csv", index=False)
        bus_stops.to_csv(self.output_dir / "candidate_bus_stops.csv", index=False)
        od_demand.to_csv(self.output_dir / "od_demand.csv", index=False)

        print("-" * 50)
        print(f"All datasets saved to {self.output_dir}/")

        return {
            "population_grid": grid,
            "metro_stations": stations,
            "bus_stops": bus_stops,
            "od_demand": od_demand,
        }

