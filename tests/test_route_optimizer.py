import math
import numpy as np
import pandas as pd
import pytest

from src.route_optimizer import great_circle_km, RouteOptimizer


# --- great_circle_km tests ---

def test_great_circle_same_point():
    """Zero distance for identical coords."""
    assert great_circle_km(24.7136, 46.6753, 24.7136, 46.6753) == 0.0


def test_great_circle_known_distance():
    # Riyadh to Jeddah, roughly 850 km
    dist = great_circle_km(24.7136, 46.6753, 21.4858, 39.1925)
    assert 840 < dist < 870


def test_great_circle_short_distance():
    # ~1 degree lat is about 111 km
    dist = great_circle_km(24.0, 46.0, 25.0, 46.0)
    assert 110 < dist < 112


def test_great_circle_symmetry():
    d1 = great_circle_km(24.7, 46.6, 25.0, 47.0)
    d2 = great_circle_km(25.0, 47.0, 24.7, 46.6)
    assert abs(d1 - d2) < 1e-9


def test_great_circle_positive():
    dist = great_circle_km(10.0, 20.0, -10.0, -20.0)
    assert dist > 0


def test_great_circle_antipodal():
    dist = great_circle_km(0, 0, 0, 180)
    assert abs(dist - math.pi * 6371.0) < 1.0


# --- RouteOptimizer init tests ---

def test_optimizer_defaults():
    opt = RouteOptimizer()
    assert opt.max_routes == 45
    assert opt.max_stops_per_route == 12
    assert opt.fleet_size == 90
    assert opt.seed == 42


def test_optimizer_custom_params():
    opt = RouteOptimizer(max_routes=10, max_stops_per_route=5, fleet_size=20, seed=99)
    assert opt.max_routes == 10
    assert opt.fleet_size == 20


# --- Helper to build minimal test data ---

def _make_test_data(n_stops=10):
    rng = np.random.default_rng(0)
    bus_stops = pd.DataFrame({
        "stop_id": range(n_stops),
        "latitude": 24.6 + rng.uniform(0, 0.2, n_stops),
        "longitude": 46.6 + rng.uniform(0, 0.2, n_stops),
        "surrounding_density": rng.uniform(1000, 5000, n_stops),
        "commercial_activity": rng.uniform(0, 1, n_stops),
        "walkability_score": rng.uniform(0, 1, n_stops),
    })
    metro = pd.DataFrame({
        "station_id": [0, 1],
        "latitude": [24.7, 24.75],
        "longitude": [46.7, 46.75],
        "line": ["Line 1", "Line 2"],
    })
    pop_grid = pd.DataFrame({
        "latitude": [24.65, 24.70, 24.75],
        "longitude": [46.65, 46.70, 46.75],
        "population_density": [5000, 8000, 3000],
        "commercial_activity": [0.5, 0.7, 0.3],
        "walkability_score": [0.6, 0.8, 0.4],
        "avg_income": [8000, 12000, 6000],
    })
    demand = rng.uniform(100, 1000, n_stops).astype(float)
    return bus_stops, metro, pop_grid, demand


# --- setup tests ---

def test_setup_stores_data():
    opt = RouteOptimizer()
    bus_stops, metro, pop_grid, demand = _make_test_data()
    opt.setup(bus_stops, metro, pop_grid, demand)
    assert opt.bus_stops is bus_stops
    assert opt.metro_stations is metro
    assert len(opt.demand_predictions) == 10


# --- _route_travel_time tests ---

def test_travel_time_single_stop():
    opt = RouteOptimizer()
    bus_stops, metro, pop_grid, demand = _make_test_data()
    opt.setup(bus_stops, metro, pop_grid, demand)
    assert opt._route_travel_time([0]) == 0.0


def test_travel_time_two_stops():
    opt = RouteOptimizer()
    bus_stops, metro, pop_grid, demand = _make_test_data()
    opt.setup(bus_stops, metro, pop_grid, demand)
    tt = opt._route_travel_time([0, 1])
    assert tt > 0


def test_travel_time_increases_with_stops():
    opt = RouteOptimizer()
    bus_stops, metro, pop_grid, demand = _make_test_data()
    opt.setup(bus_stops, metro, pop_grid, demand)
    t2 = opt._route_travel_time([0, 1])
    t3 = opt._route_travel_time([0, 1, 2])
    assert t3 >= t2


# --- _coverage_score tests ---

def test_coverage_empty_routes():
    opt = RouteOptimizer()
    bus_stops, metro, pop_grid, demand = _make_test_data()
    opt.setup(bus_stops, metro, pop_grid, demand)
    assert opt._coverage_score([]) == 0.0


def test_coverage_with_stops():
    opt = RouteOptimizer()
    bus_stops, metro, pop_grid, demand = _make_test_data()
    opt.setup(bus_stops, metro, pop_grid, demand)
    score = opt._coverage_score([[0, 1, 2, 3, 4]])
    assert 0.0 <= score <= 1.0


# --- _fitness tests ---

def test_fitness_empty_gives_zero():
    opt = RouteOptimizer(max_stops_per_route=4)
    bus_stops, metro, pop_grid, demand = _make_test_data()
    opt.setup(bus_stops, metro, pop_grid, demand)
    # all -1 means no valid stops
    individual = [-1] * (opt.max_routes * opt.max_stops_per_route)
    result = opt._fitness(individual)
    assert result == (0.0,)


def test_fitness_returns_tuple():
    opt = RouteOptimizer(max_routes=2, max_stops_per_route=4)
    bus_stops, metro, pop_grid, demand = _make_test_data()
    opt.setup(bus_stops, metro, pop_grid, demand)
    individual = [0, 1, 2, 3, 4, 5, 6, 7]
    result = opt._fitness(individual)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0] >= 0.0


# --- get_convergence_df ---

def test_convergence_df_empty():
    opt = RouteOptimizer()
    df = opt.get_convergence_df()
    assert len(df) == 0


# --- get_route_summary ---

def test_route_summary_empty():
    opt = RouteOptimizer()
    bus_stops, metro, pop_grid, demand = _make_test_data()
    opt.setup(bus_stops, metro, pop_grid, demand)
    df = opt.get_route_summary(routes=[])
    assert len(df) == 0


def test_route_summary_with_routes():
    opt = RouteOptimizer()
    bus_stops, metro, pop_grid, demand = _make_test_data()
    opt.setup(bus_stops, metro, pop_grid, demand)
    df = opt.get_route_summary(routes=[[0, 1, 2], [3, 4]])
    assert len(df) == 2
    assert "total_distance_km" in df.columns
    assert "travel_time_min" in df.columns
    assert all(df["total_distance_km"] >= 0)


# --- greedy_baseline ---

def test_greedy_baseline_returns_routes():
    opt = RouteOptimizer(max_routes=3, max_stops_per_route=4, seed=42)
    bus_stops, metro, pop_grid, demand = _make_test_data(20)
    opt.setup(bus_stops, metro, pop_grid, demand)
    routes = opt.greedy_baseline()
    assert isinstance(routes, list)
    assert len(routes) > 0
    for r in routes:
        assert len(r) >= 2


def test_greedy_no_duplicate_stops_across_routes():
    opt = RouteOptimizer(max_routes=5, max_stops_per_route=4, seed=42)
    bus_stops, metro, pop_grid, demand = _make_test_data(30)
    opt.setup(bus_stops, metro, pop_grid, demand)
    routes = opt.greedy_baseline()
    all_stops = []
    for r in routes:
        all_stops.extend(r)
    assert len(all_stops) == len(set(all_stops))
