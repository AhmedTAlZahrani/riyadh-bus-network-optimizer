import numpy as np
import pandas as pd
from pathlib import Path
import json


def _haversine(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


class RouteEvaluator:

    def __init__(self, bus_stops, metro_stations, population_grid):
        """Initialize the evaluator with transit data.

        Args:
            bus_stops: DataFrame of bus stop locations.
            metro_stations: DataFrame of metro station locations.
            population_grid: DataFrame of population density grid.
        """
        self.bus_stops = bus_stops
        self.metro_stations = metro_stations
        self.population_grid = population_grid

    def coverage_metrics(self, routes, radius_km=0.4):
        """Calculate population coverage within a radius of bus stops.

        Args:
            routes: List of routes (each route is a list of stop IDs).
            radius_km: Walking distance threshold in km.

        Returns:
            Dict with coverage statistics.
        """
        all_stop_ids = set()
        for route in routes:
            all_stop_ids.update(route)

        stop_coords = []
        for sid in all_stop_ids:
            row = self.bus_stops[self.bus_stops["stop_id"] == sid]
            if len(row) > 0:
                stop_coords.append((row.iloc[0]["latitude"], row.iloc[0]["longitude"]))

        covered_cells = 0
        covered_pop = 0
        total_pop = 0

        for _, cell in self.population_grid.iterrows():
            total_pop += cell["population_density"]
            for slat, slon in stop_coords:
                dist = _haversine(cell["latitude"], cell["longitude"], slat, slon)
                if dist <= radius_km:
                    covered_cells += 1
                    covered_pop += cell["population_density"]
                    break

        total_cells = len(self.population_grid)
        coverage_pct = (covered_pop / max(total_pop, 1)) * 100
        cell_coverage_pct = (covered_cells / max(total_cells, 1)) * 100

        metrics = {
            "total_stops_used": len(all_stop_ids),
            "total_routes": len(routes),
            "coverage_radius_km": radius_km,
            "population_coverage_pct": round(coverage_pct, 1),
            "cell_coverage_pct": round(cell_coverage_pct, 1),
            "covered_population": round(covered_pop, 0),
            "total_population": round(total_pop, 0),
        }

        print(f"Coverage ({radius_km}km): {coverage_pct:.1f}% population, "
              f"{cell_coverage_pct:.1f}% cells")
        return metrics

    def transfer_time_distribution(self, routes):
        """Calculate bus-to-metro transfer time distribution.

        Estimates time from each bus stop to the nearest metro station
        assuming walking speed of 5 km/h plus 2 minutes wait/transfer.

        Args:
            routes: List of routes (each route is a list of stop IDs).

        Returns:
            DataFrame with transfer times per stop.
        """
        all_stop_ids = set()
        for route in routes:
            all_stop_ids.update(route)

        station_lats = self.metro_stations["latitude"].values
        station_lons = self.metro_stations["longitude"].values

        rows = []
        for sid in all_stop_ids:
            stop_row = self.bus_stops[self.bus_stops["stop_id"] == sid]
            if len(stop_row) == 0:
                continue

            slat = stop_row.iloc[0]["latitude"]
            slon = stop_row.iloc[0]["longitude"]

            distances = np.array([
                _haversine(slat, slon, mlat, mlon)
                for mlat, mlon in zip(station_lats, station_lons)
            ])

            nearest_dist = distances.min()
            nearest_station = int(np.argmin(distances))

            walk_time = (nearest_dist / 5.0) * 60  # minutes at 5 km/h
            transfer_time = walk_time + 2.0  # 2 min transfer overhead

            rows.append({
                "stop_id": sid,
                "nearest_station_id": nearest_station,
                "distance_km": round(nearest_dist, 3),
                "walk_time_min": round(walk_time, 1),
                "transfer_time_min": round(transfer_time, 1),
            })

        df = pd.DataFrame(rows)

        if len(df) > 0:
            print(f"Transfer times: mean={df['transfer_time_min'].mean():.1f} min, "
                  f"median={df['transfer_time_min'].median():.1f} min, "
                  f"max={df['transfer_time_min'].max():.1f} min")

        return df

    def cost_per_rider(self, routes, demand_predictions, operating_cost_per_km=8.0):
        """Estimate cost per rider for the bus network.

        Args:
            routes: List of routes.
            demand_predictions: Array of predicted demand per stop.
            operating_cost_per_km: Operating cost in SAR per km.

        Returns:
            Dict with cost estimates.
        """
        total_distance = 0
        total_riders = 0

        for route in routes:
            for i in range(len(route) - 1):
                s1 = self.bus_stops[self.bus_stops["stop_id"] == route[i]]
                s2 = self.bus_stops[self.bus_stops["stop_id"] == route[i + 1]]
                if len(s1) > 0 and len(s2) > 0:
                    total_distance += _haversine(
                        s1.iloc[0]["latitude"], s1.iloc[0]["longitude"],
                        s2.iloc[0]["latitude"], s2.iloc[0]["longitude"],
                    )

            for sid in route:
                idx = self.bus_stops[self.bus_stops["stop_id"] == sid].index
                if len(idx) > 0 and idx[0] < len(demand_predictions):
                    total_riders += demand_predictions[idx[0]]

        daily_cost = total_distance * 2 * operating_cost_per_km  # round trips
        cost_rider = daily_cost / max(total_riders, 1)

        costs = {
            "total_route_distance_km": round(total_distance, 1),
            "daily_operating_cost_sar": round(daily_cost, 0),
            "estimated_daily_riders": round(total_riders, 0),
            "cost_per_rider_sar": round(cost_rider, 2),
        }

        print(f"Cost per rider: {cost_rider:.2f} SAR "
              f"({total_riders:.0f} riders, {daily_cost:.0f} SAR/day)")
        return costs

    def before_after_comparison(self, optimized_routes, greedy_routes, demand_predictions):
        """Compare optimized routes against greedy baseline.

        Args:
            optimized_routes: Routes from genetic algorithm.
            greedy_routes: Routes from greedy baseline.
            demand_predictions: Array of predicted demand per stop.

        Returns:
            DataFrame comparing before and after metrics.
        """
        print("\n--- Before/After Comparison ---")

        print("\nGreedy baseline:")
        greedy_coverage = self.coverage_metrics(greedy_routes)
        greedy_transfer = self.transfer_time_distribution(greedy_routes)
        greedy_cost = self.cost_per_rider(greedy_routes, demand_predictions)

        print("\nOptimized routes:")
        opt_coverage = self.coverage_metrics(optimized_routes)
        opt_transfer = self.transfer_time_distribution(optimized_routes)
        opt_cost = self.cost_per_rider(optimized_routes, demand_predictions)

        comparison = pd.DataFrame([
            {
                "Metric": "Population Coverage (400m)",
                "Before (Greedy)": f"{greedy_coverage['population_coverage_pct']:.1f}%",
                "After (Optimized)": f"{opt_coverage['population_coverage_pct']:.1f}%",
            },
            {
                "Metric": "Avg Transfer Time (min)",
                "Before (Greedy)": f"{greedy_transfer['transfer_time_min'].mean():.1f}"
                    if len(greedy_transfer) > 0 else "N/A",
                "After (Optimized)": f"{opt_transfer['transfer_time_min'].mean():.1f}"
                    if len(opt_transfer) > 0 else "N/A",
            },
            {
                "Metric": "Routes Used",
                "Before (Greedy)": str(len(greedy_routes)),
                "After (Optimized)": str(len(optimized_routes)),
            },
            {
                "Metric": "Cost per Rider (SAR)",
                "Before (Greedy)": f"{greedy_cost['cost_per_rider_sar']:.2f}",
                "After (Optimized)": f"{opt_cost['cost_per_rider_sar']:.2f}",
            },
            {
                "Metric": "Total Stops",
                "Before (Greedy)": str(greedy_coverage["total_stops_used"]),
                "After (Optimized)": str(opt_coverage["total_stops_used"]),
            },
        ])

        return comparison

    def route_quality_scores(self, routes, demand_predictions):
        """Score individual routes on multiple quality dimensions.

        Args:
            routes: List of routes.
            demand_predictions: Array of predicted demand per stop.

        Returns:
            DataFrame with quality scores per route.
        """
        rows = []

        for idx, route in enumerate(routes):
            if len(route) < 2:
                continue

            # Route distance
            total_dist = 0
            for i in range(len(route) - 1):
                s1 = self.bus_stops[self.bus_stops["stop_id"] == route[i]]
                s2 = self.bus_stops[self.bus_stops["stop_id"] == route[i + 1]]
                if len(s1) > 0 and len(s2) > 0:
                    total_dist += _haversine(
                        s1.iloc[0]["latitude"], s1.iloc[0]["longitude"],
                        s2.iloc[0]["latitude"], s2.iloc[0]["longitude"],
                    )

            # Demand served
            route_demand = 0
            for sid in route:
                idx_arr = self.bus_stops[self.bus_stops["stop_id"] == sid].index
                if len(idx_arr) > 0 and idx_arr[0] < len(demand_predictions):
                    route_demand += demand_predictions[idx_arr[0]]

            # Directness
            first = self.bus_stops[self.bus_stops["stop_id"] == route[0]]
            last = self.bus_stops[self.bus_stops["stop_id"] == route[-1]]
            if len(first) > 0 and len(last) > 0:
                straight = _haversine(
                    first.iloc[0]["latitude"], first.iloc[0]["longitude"],
                    last.iloc[0]["latitude"], last.iloc[0]["longitude"],
                )
                directness = straight / max(total_dist, 0.01)
            else:
                directness = 0

            # Demand per km
            demand_per_km = route_demand / max(total_dist, 0.01)

            # Overall quality score (0-100)
            quality = min(100, (
                30 * min(directness, 1.0)
                + 40 * min(demand_per_km / 2000, 1.0)
                + 30 * min(len(route) / self.max_stops_per_route, 1.0)
            ))

            rows.append({
                "route_id": idx,
                "n_stops": len(route),
                "distance_km": round(total_dist, 2),
                "demand_served": round(route_demand, 0),
                "directness": round(directness, 3),
                "demand_per_km": round(demand_per_km, 1),
                "quality_score": round(quality, 1),
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            print(f"Route quality: mean={df['quality_score'].mean():.1f}, "
                  f"min={df['quality_score'].min():.1f}, "
                  f"max={df['quality_score'].max():.1f}")
        return df

    def save_results(self, results_dict, path="output/evaluation_results.json"):
        """Save evaluation results to a JSON file.

        Args:
            results_dict: Dict of evaluation metrics.
            path: Output file path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
        print(f"Evaluation results saved to {output}")

