import math
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random


def great_circle_km(lat1, lon1, lat2, lon2):
    """Distance between two lat/lon points in km using the haversine formula."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class RouteOptimizer:

    def __init__(self, max_routes=45, max_stops_per_route=12, fleet_size=90, seed=42):
        self.max_routes = max_routes
        self.max_stops_per_route = max_stops_per_route
        self.fleet_size = fleet_size
        self.seed = seed

        self.bus_stops = None
        self.metro_stations = None
        self.population_grid = None
        self.demand_predictions = None
        self.best_routes = None
        self.convergence_history = []

    def setup(self, bus_stops, metro_stations, population_grid, demand_predictions):
        """Configure the optimizer with transit data."""
        self.bus_stops = bus_stops
        self.metro_stations = metro_stations
        self.population_grid = population_grid
        self.demand_predictions = demand_predictions
        print(f"Optimizer configured: {len(bus_stops)} stops, "
              f"{self.max_routes} routes, fleet={self.fleet_size}")

    def _coverage_score(self, route_stop_ids, radius_km=0.4):
        all_stops = set()
        for route in route_stop_ids:
            all_stops.update(route)

        if not all_stops:
            return 0.0

        stop_lats = []
        stop_lons = []
        for sid in all_stops:
            row = self.bus_stops[self.bus_stops["stop_id"] == sid]
            if len(row) > 0:
                stop_lats.append(row.iloc[0]["latitude"])
                stop_lons.append(row.iloc[0]["longitude"])

        covered_pop = 0
        total_pop = self.population_grid["population_density"].sum()

        for _, cell in self.population_grid.iterrows():
            for slat, slon in zip(stop_lats, stop_lons):
                dist = great_circle_km(cell["latitude"], cell["longitude"], slat, slon)
                if dist <= radius_km:
                    covered_pop += cell["population_density"]
                    break

        return covered_pop / max(total_pop, 1)

    def _route_travel_time(self, stop_ids):
        if len(stop_ids) < 2:
            return 0.0

        total_dist = 0
        for i in range(len(stop_ids) - 1):
            s1 = self.bus_stops[self.bus_stops["stop_id"] == stop_ids[i]]
            s2 = self.bus_stops[self.bus_stops["stop_id"] == stop_ids[i + 1]]
            if len(s1) > 0 and len(s2) > 0:
                total_dist += great_circle_km(
                    s1.iloc[0]["latitude"], s1.iloc[0]["longitude"],
                    s2.iloc[0]["latitude"], s2.iloc[0]["longitude"],
                )

        return (total_dist / 30.0) * 60  # 30 km/h average speed

    def _fitness(self, individual):
        # Decode individual into routes
        routes = []
        for i in range(0, len(individual), self.max_stops_per_route):
            route = [s for s in individual[i:i + self.max_stops_per_route] if s >= 0]
            if len(route) >= 2:
                routes.append(route)

        if not routes:
            return (0.0,)

        # Coverage score (weight: 0.5)
        coverage = self._coverage_score(routes)

        # Travel time penalty (weight: 0.3)
        total_time = sum(self._route_travel_time(r) for r in routes)
        max_time = self.max_routes * self.max_stops_per_route * 5  # rough upper bound
        time_score = 1.0 - min(total_time / max(max_time, 1), 1.0)

        # Demand score (weight: 0.2)
        all_stops = set()
        for route in routes:
            all_stops.update(route)
        demand_sum = 0
        for sid in all_stops:
            idx = self.bus_stops[self.bus_stops["stop_id"] == sid].index
            if len(idx) > 0 and idx[0] < len(self.demand_predictions):
                demand_sum += self.demand_predictions[idx[0]]
        max_demand = np.sum(self.demand_predictions)
        demand_score = demand_sum / max(max_demand, 1)

        # Fleet constraint penalty
        fleet_penalty = 0
        if len(routes) > self.fleet_size:
            fleet_penalty = 0.2 * (len(routes) - self.fleet_size) / self.fleet_size

        fitness = 0.5 * coverage + 0.3 * time_score + 0.2 * demand_score - fleet_penalty

        return (max(fitness, 0.0),)

    # FIXME: breaks when n_clusters > n_samples
    def optimize(self, n_generations=100, population_size=50):
        """Run the genetic algorithm to find optimal bus routes."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_stops = len(self.bus_stops)
        chromosome_length = self.max_routes * self.max_stops_per_route

        # DEAP setup
        if "FitnessMax" not in dir(creator):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if "Individual" not in dir(creator):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("gene", random.randint, -1, n_stops - 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.gene, n=chromosome_length)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._fitness)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=-1, up=n_stops - 1,
                         indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        print(f"Starting genetic algorithm: {n_generations} generations, "
              f"pop={population_size}")

        pop = toolbox.population(n=population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("max", np.max)
        stats.register("avg", np.mean)

        self.convergence_history = []

        for gen in range(n_generations):
            # Evaluate
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            # Record stats
            record = stats.compile(pop)
            self.convergence_history.append({
                "generation": gen,
                "max_fitness": round(record["max"], 4),
                "avg_fitness": round(record["avg"], 4),
            })

            if gen % 20 == 0:
                print(f"  Gen {gen:3d}: max={record['max']:.4f}, avg={record['avg']:.4f}")

            # Select and breed
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            pop[:] = offspring

        # Final evaluation
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        best_individual = tools.selBest(pop, 1)[0]
        best_fitness = best_individual.fitness.values[0]
        print(f"Optimization complete: best fitness = {best_fitness:.4f}")

        # Decode best individual into routes
        print(f"Optimizing {len(self.best_routes) if self.best_routes else 0} routes...")
        self.best_routes = []
        for i in range(0, len(best_individual), self.max_stops_per_route):
            route = [s for s in best_individual[i:i + self.max_stops_per_route] if s >= 0]
            if len(route) >= 2:
                # Remove duplicate stops
                seen = set()
                unique_route = []
                for s in route:
                    if s not in seen:
                        seen.add(s)
                        unique_route.append(s)
                if len(unique_route) >= 2:
                    self.best_routes.append(unique_route)

        print(f"Generated {len(self.best_routes)} routes with "
              f"{sum(len(r) for r in self.best_routes)} total stops")

        return self.best_routes

    def greedy_baseline(self):
        """Generate routes using a greedy nearest-neighbor heuristic."""
        print("Generating greedy baseline routes...")
        rng = np.random.default_rng(self.seed)

        # Sort stops by demand
        stop_demand = list(zip(self.bus_stops["stop_id"].values, self.demand_predictions))
        stop_demand.sort(key=lambda x: x[1], reverse=True)

        covered = set()
        routes = []

        for _ in range(self.max_routes):
            # Find seed: highest demand uncovered stop
            seed = None
            for sid, _ in stop_demand:
                if sid not in covered:
                    seed = sid
                    break
            if seed is None:
                break

            route = [seed]
            covered.add(seed)

            # Extend route greedily
            for _ in range(self.max_stops_per_route - 1):
                current = route[-1]
                current_row = self.bus_stops[self.bus_stops["stop_id"] == current].iloc[0]

                best_next = None
                best_score = -1

                for sid, demand in stop_demand:
                    if sid in covered or sid in route:
                        continue
                    row = self.bus_stops[self.bus_stops["stop_id"] == sid].iloc[0]
                    dist = great_circle_km(current_row["latitude"], current_row["longitude"],
                                      row["latitude"], row["longitude"])

                    if dist > 3.0:  # Max 3 km between consecutive stops
                        continue

                    score = demand / max(dist, 0.1)
                    if score > best_score:
                        best_score = score
                        best_next = sid

                if best_next is None:
                    break

                route.append(best_next)
                covered.add(best_next)

            if len(route) >= 2:
                routes.append(route)

        print(f"Greedy baseline: {len(routes)} routes, "
              f"{sum(len(r) for r in routes)} total stops")
        return routes

    def get_convergence_df(self):
        """Return convergence history as a DataFrame."""
        return pd.DataFrame(self.convergence_history)

    def get_route_summary(self, routes=None):
        """Generate summary statistics for a set of routes."""
        routes = routes or self.best_routes or []
        rows = []

        for idx, route in enumerate(routes):
            total_dist = 0
            for i in range(len(route) - 1):
                s1 = self.bus_stops[self.bus_stops["stop_id"] == route[i]]
                s2 = self.bus_stops[self.bus_stops["stop_id"] == route[i + 1]]
                if len(s1) > 0 and len(s2) > 0:
                    total_dist += great_circle_km(
                        s1.iloc[0]["latitude"], s1.iloc[0]["longitude"],
                        s2.iloc[0]["latitude"], s2.iloc[0]["longitude"],
                    )

            route_demand = 0
            for sid in route:
                idx_arr = self.bus_stops[self.bus_stops["stop_id"] == sid].index
                if len(idx_arr) > 0 and idx_arr[0] < len(self.demand_predictions):
                    route_demand += self.demand_predictions[idx_arr[0]]

            rows.append({
                "route_id": idx,
                "n_stops": len(route),
                "total_distance_km": round(total_dist, 2),
                "travel_time_min": round((total_dist / 30.0) * 60, 1),
                "total_demand": round(route_demand, 0),
            })

        return pd.DataFrame(rows)

