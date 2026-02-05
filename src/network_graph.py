import numpy as np
import pandas as pd
import networkx as nx


def _haversine(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# Major road corridors in Riyadh (approximate)
MAJOR_ROADS = [
    {"name": "King Fahd Road", "points": [(24.90, 46.685), (24.80, 46.680), (24.70, 46.675), (24.60, 46.670)]},
    {"name": "Olaya Street", "points": [(24.85, 46.695), (24.75, 46.690), (24.65, 46.685)]},
    {"name": "Eastern Ring", "points": [(24.85, 46.780), (24.75, 46.775), (24.65, 46.770)]},
    {"name": "Northern Ring", "points": [(24.82, 46.600), (24.82, 46.650), (24.82, 46.700), (24.82, 46.750)]},
    {"name": "Southern Ring", "points": [(24.60, 46.600), (24.60, 46.650), (24.60, 46.700), (24.60, 46.750)]},
    {"name": "Makkah Road", "points": [(24.65, 46.580), (24.65, 46.640), (24.65, 46.700), (24.65, 46.760)]},
    {"name": "Khurais Road", "points": [(24.72, 46.600), (24.72, 46.660), (24.72, 46.720), (24.72, 46.780)]},
]

AVG_SPEED_KMH = 30.0  # Average urban bus speed


class TransitNetwork:

    def __init__(self, grid_spacing=0.01):
        """Initialize the transit network.

        Args:
            grid_spacing: Spacing between grid nodes in degrees.
        """
        self.graph = nx.Graph()
        self.grid_spacing = grid_spacing
        self._node_positions = {}

    def build_network(self, bus_stops, metro_stations):
        """Build the road network graph with grid and major roads.

        Args:
            bus_stops: DataFrame of candidate bus stops.
            metro_stations: DataFrame of metro station locations.

        Returns:
            The constructed NetworkX graph.
        """
        print("Building transit network graph...")

        # Add grid nodes
        lat_range = np.arange(24.55, 24.95, self.grid_spacing)
        lon_range = np.arange(46.55, 46.85, self.grid_spacing)

        for lat in lat_range:
            for lon in lon_range:
                node_id = f"grid_{lat:.3f}_{lon:.3f}"
                self.graph.add_node(node_id, lat=lat, lon=lon, node_type="grid")
                self._node_positions[node_id] = (lat, lon)

        # Connect grid neighbors
        grid_nodes = [n for n in self.graph.nodes if self.graph.nodes[n].get("node_type") == "grid"]
        for i, n1 in enumerate(grid_nodes):
            lat1, lon1 = self._node_positions[n1]
            for n2 in grid_nodes[i + 1:]:
                lat2, lon2 = self._node_positions[n2]
                if abs(lat1 - lat2) <= self.grid_spacing * 1.1 and abs(lon1 - lon2) <= self.grid_spacing * 1.1:
                    dist = _haversine(lat1, lon1, lat2, lon2)
                    travel_time = (dist / AVG_SPEED_KMH) * 60  # minutes
                    self.graph.add_edge(n1, n2, weight=dist, travel_time=travel_time)

        # Add major road connections with lower weight
        for road in MAJOR_ROADS:
            points = road["points"]
            for k in range(len(points) - 1):
                lat1, lon1 = points[k]
                lat2, lon2 = points[k + 1]
                n1 = self._find_nearest_node(lat1, lon1)
                n2 = self._find_nearest_node(lat2, lon2)
                if n1 and n2:
                    dist = _haversine(lat1, lon1, lat2, lon2)
                    fast_time = (dist / (AVG_SPEED_KMH * 1.5)) * 60
                    self.graph.add_edge(n1, n2, weight=dist, travel_time=fast_time, road=road["name"])

        # Add bus stop nodes
        for _, stop in bus_stops.iterrows():
            node_id = f"bus_{stop['stop_id']}"
            self.graph.add_node(node_id, lat=stop["latitude"], lon=stop["longitude"], node_type="bus_stop")
            self._node_positions[node_id] = (stop["latitude"], stop["longitude"])
            nearest = self._find_nearest_node(stop["latitude"], stop["longitude"], exclude_prefix="bus_")
            if nearest:
                dist = _haversine(stop["latitude"], stop["longitude"],
                                  *self._node_positions[nearest])
                self.graph.add_edge(node_id, nearest, weight=dist,
                                    travel_time=(dist / AVG_SPEED_KMH) * 60)

        # Add metro station nodes
        for _, station in metro_stations.iterrows():
            node_id = f"metro_{station['station_id']}"
            self.graph.add_node(node_id, lat=station["latitude"], lon=station["longitude"],
                                node_type="metro_station", line=station["line"])
            self._node_positions[node_id] = (station["latitude"], station["longitude"])
            nearest = self._find_nearest_node(station["latitude"], station["longitude"],
                                              exclude_prefix="metro_")
            if nearest:
                dist = _haversine(station["latitude"], station["longitude"],
                                  *self._node_positions[nearest])
                self.graph.add_edge(node_id, nearest, weight=dist,
                                    travel_time=(dist / AVG_SPEED_KMH) * 60)

        print(f"Network built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        return self.graph

    def _find_nearest_node(self, lat, lon, exclude_prefix=None):
        """Find the nearest node in the graph to a coordinate.

        Args:
            lat: Target latitude.
            lon: Target longitude.
            exclude_prefix: Skip nodes whose ID starts with this prefix.

        Returns:
            Node ID of nearest node, or None.
        """
        best_node = None
        best_dist = float("inf")

        for node_id, (nlat, nlon) in self._node_positions.items():
            if exclude_prefix and node_id.startswith(exclude_prefix):
                continue
            dist = _haversine(lat, lon, nlat, nlon)
            if dist < best_dist:
                best_dist = dist
                best_node = node_id

        return best_node

    def shortest_path(self, origin_id, dest_id, weight="travel_time"):
        """Calculate shortest path between two nodes.

        Args:
            origin_id: Source node ID.
            dest_id: Destination node ID.
            weight: Edge attribute to minimize.

        Returns:
            Tuple of (path as list of node IDs, total cost).
        """
        try:
            path = nx.shortest_path(self.graph, origin_id, dest_id, weight=weight)
            cost = nx.shortest_path_length(self.graph, origin_id, dest_id, weight=weight)
            return path, round(cost, 2)
        except nx.NetworkXNoPath:
            return [], float("inf")

    def route_travel_time(self, stop_ids):
        """Calculate total travel time for a bus route visiting stops in order.

        Args:
            stop_ids: List of bus stop IDs forming the route.

        Returns:
            Total travel time in minutes.
        """
        total_time = 0
        for i in range(len(stop_ids) - 1):
            origin = f"bus_{stop_ids[i]}"
            dest = f"bus_{stop_ids[i + 1]}"
            _, time = self.shortest_path(origin, dest)
            total_time += time

        return round(total_time, 2)

    def route_feasibility_score(self, stop_ids, metro_stations):
        """Score a route based on directness, coverage, and travel time.

        Args:
            stop_ids: List of bus stop IDs in the route.
            metro_stations: DataFrame of metro stations.

        Returns:
            Dict with directness, coverage_radius, and total_time scores.
        """
        if len(stop_ids) < 2:
            return {"directness": 0, "coverage_radius_km": 0, "total_time_min": 0}

        # Travel time
        total_time = self.route_travel_time(stop_ids)

        # Directness: ratio of straight-line to route distance
        first_node = f"bus_{stop_ids[0]}"
        last_node = f"bus_{stop_ids[-1]}"
        first_pos = self._node_positions.get(first_node, (0, 0))
        last_pos = self._node_positions.get(last_node, (0, 0))
        straight_dist = _haversine(first_pos[0], first_pos[1], last_pos[0], last_pos[1])

        route_dist = 0
        for i in range(len(stop_ids) - 1):
            n1 = f"bus_{stop_ids[i]}"
            n2 = f"bus_{stop_ids[i + 1]}"
            p1 = self._node_positions.get(n1, (0, 0))
            p2 = self._node_positions.get(n2, (0, 0))
            route_dist += _haversine(p1[0], p1[1], p2[0], p2[1])

        directness = straight_dist / max(route_dist, 0.01)

        # Coverage radius
        lats = [self._node_positions.get(f"bus_{s}", (0, 0))[0] for s in stop_ids]
        lons = [self._node_positions.get(f"bus_{s}", (0, 0))[1] for s in stop_ids]
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        max_radius = max(
            _haversine(center_lat, center_lon, lat, lon)
            for lat, lon in zip(lats, lons)
        )

        return {
            "directness": round(directness, 3),
            "coverage_radius_km": round(max_radius, 2),
            "total_time_min": total_time,
        }

    def connectivity_metrics(self):
        """Calculate network connectivity metrics.

        Returns:
            Dict with graph statistics.
        """
        bus_nodes = [n for n in self.graph.nodes if str(n).startswith("bus_")]
        metro_nodes = [n for n in self.graph.nodes if str(n).startswith("metro_")]

        # Check connectivity for bus-metro pairs
        connected_pairs = 0
        total_pairs = 0
        for bus in bus_nodes[:50]:  # Sample for efficiency
            for metro in metro_nodes[:20]:
                total_pairs += 1
                if nx.has_path(self.graph, bus, metro):
                    connected_pairs += 1

        connectivity_rate = connected_pairs / max(total_pairs, 1)

        metrics = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "bus_stops": len(bus_nodes),
            "metro_stations": len(metro_nodes),
            "avg_degree": round(np.mean([d for _, d in self.graph.degree()]), 2),
            "bus_metro_connectivity": round(connectivity_rate, 3),
        }

        print(f"Network connectivity: {metrics['bus_metro_connectivity']:.1%} "
              f"bus-metro pairs connected")
        return metrics

    def get_node_positions(self):
        """Return node positions as a dict mapping node ID to (lat, lon).

        Returns:
            Dict of node positions.
        """
        return dict(self._node_positions)

