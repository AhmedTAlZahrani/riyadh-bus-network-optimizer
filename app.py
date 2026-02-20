import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

from src.data_generator import RiyadhDataGenerator, METRO_LINES, RIYADH_LAT, RIYADH_LON
from src.demand_model import DemandPredictor
from src.network_graph import TransitNetwork
from src.route_optimizer import RouteOptimizer
from src.evaluation import RouteEvaluator


st.set_page_config(page_title="Riyadh Bus Network Optimizer", layout="wide")
st.title("Riyadh Bus Network Optimizer")
st.markdown("*Graph-based feeder bus route optimization for Riyadh Metro's last-mile connectivity*")

# --- Sidebar ---
st.sidebar.header("Optimization Parameters")
fleet_size = st.sidebar.slider("Fleet Size", 20, 150, 90, step=10)
max_stops = st.sidebar.slider("Max Stops per Route", 4, 20, 12)
n_generations = st.sidebar.slider("GA Generations", 20, 200, 80, step=10)
max_routes = st.sidebar.slider("Max Routes", 10, 80, 45, step=5)

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Optimization", type="primary")


@st.cache_data
def load_data():
    """Generate and cache transit network data."""
    gen = RiyadhDataGenerator(seed=42, output_dir="data")
    return gen.generate_all()


@st.cache_data
def train_demand_model(_data):
    """Train and cache the demand prediction model."""
    predictor = DemandPredictor(output_dir="models")
    features = predictor.prepare_features(
        _data["bus_stops"], _data["metro_stations"], _data["population_grid"]
    )
    demand = predictor.generate_demand_labels(features)
    predictor.train(features, demand)
    predictions = predictor.predict(features)
    importance = predictor.get_feature_importance()
    return predictor, features, demand, predictions, importance


data = load_data()
predictor, features, demand, predictions, importance = train_demand_model(data)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Population & Demand",
    "Network Graph",
    "Optimized Routes",
    "Results",
])


# ============================================================
# Tab 1: Population & Demand
# ============================================================
with tab1:
    st.header("Population Density & Demand Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Population Density Heatmap")
        m = folium.Map(location=[RIYADH_LAT, RIYADH_LON], zoom_start=11, tiles="cartodbpositron")

        heat_data = data["population_grid"][["latitude", "longitude", "population_density"]].values.tolist()
        from folium.plugins import HeatMap
        HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)

        st_folium(m, width=600, height=450, key="pop_heatmap")

    with col2:
        st.subheader("Demand Prediction Heatmap")
        heatmap_df = predictor.generate_demand_heatmap(features, predictions)

        m2 = folium.Map(location=[RIYADH_LAT, RIYADH_LON], zoom_start=11, tiles="cartodbpositron")
        demand_data = heatmap_df[["latitude", "longitude", "demand"]].values.tolist()
        HeatMap(demand_data, radius=12, blur=8, max_zoom=13, gradient={
            0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"
        }).add_to(m2)
        st_folium(m2, width=600, height=450, key="demand_heatmap")

    st.subheader("Feature Importance")
    fig_imp = px.bar(
        importance, x="importance", y="feature", orientation="h",
        title="XGBoost Feature Importance for Demand Prediction",
        color="importance", color_continuous_scale="Viridis",
    )
    fig_imp.update_layout(template="plotly_dark", height=350, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Population Grid Statistics")
    grid = data["population_grid"]
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("Total Cells", f"{len(grid):,}")
    stat_col2.metric("Avg Density", f"{grid['population_density'].mean():,.0f}")
    stat_col3.metric("Max Density", f"{grid['population_density'].max():,.0f}")
    stat_col4.metric("Avg Income (SAR)", f"{grid['avg_income'].mean():,.0f}")


# ============================================================
# Tab 2: Network Graph
# ============================================================
with tab2:
    st.header("Metro Lines & Candidate Bus Stops")

    m3 = folium.Map(location=[RIYADH_LAT, RIYADH_LON], zoom_start=11, tiles="cartodbpositron")

    # Draw metro lines
    for line_name, line_info in METRO_LINES.items():
        coords = [(lat, lon) for lat, lon in line_info["stations"]]
        folium.PolyLine(coords, color=line_info["color"], weight=4,
                        opacity=0.8, popup=line_name).add_to(m3)

        for lat, lon in line_info["stations"]:
            folium.CircleMarker(
                location=[lat, lon], radius=5, color=line_info["color"],
                fill=True, fill_opacity=0.9, popup=line_name,
            ).add_to(m3)

    # Draw candidate bus stops
    for _, stop in data["bus_stops"].iterrows():
        folium.CircleMarker(
            location=[stop["latitude"], stop["longitude"]],
            radius=2, color="gray", fill=True, fill_opacity=0.4,
            popup=f"Stop {stop['stop_id']}",
        ).add_to(m3)

    st_folium(m3, width=1000, height=600, key="network_map")

    col_a, col_b = st.columns(2)
    col_a.metric("Metro Stations", len(data["metro_stations"]))
    col_b.metric("Candidate Bus Stops", len(data["bus_stops"]))

    st.subheader("Metro Lines Summary")
    line_summary = data["metro_stations"].groupby("line").size().reset_index(name="stations")
    st.dataframe(line_summary, use_container_width=True)


# ============================================================
# Tab 3: Optimized Routes
# ============================================================
with tab3:
    st.header("Optimized Feeder Bus Routes")

    if run_button or "opt_routes" in st.session_state:
        if run_button:
            with st.spinner("Running genetic algorithm optimization..."):
                optimizer = RouteOptimizer(
                    max_routes=max_routes,
                    max_stops_per_route=max_stops,
                    fleet_size=fleet_size,
                )
                optimizer.setup(data["bus_stops"], data["metro_stations"],
                                data["population_grid"], predictions)

                greedy_routes = optimizer.greedy_baseline()
                optimized_routes = optimizer.optimize(
                    n_generations=n_generations, population_size=50
                )

                st.session_state["opt_routes"] = optimized_routes
                st.session_state["greedy_routes"] = greedy_routes
                st.session_state["convergence"] = optimizer.get_convergence_df()
                st.session_state["route_summary"] = optimizer.get_route_summary()

        optimized_routes = st.session_state["opt_routes"]

        # Map with optimized routes
        m4 = folium.Map(location=[RIYADH_LAT, RIYADH_LON], zoom_start=11,
                        tiles="cartodbpositron")

        # Metro lines (background)
        for line_name, line_info in METRO_LINES.items():
            coords = [(lat, lon) for lat, lon in line_info["stations"]]
            folium.PolyLine(coords, color=line_info["color"], weight=3,
                            opacity=0.5).add_to(m4)

        # Optimized bus routes
        route_colors = px.colors.qualitative.Set3
        for ridx, route in enumerate(optimized_routes):
            color = route_colors[ridx % len(route_colors)]
            coords = []
            for sid in route:
                row = data["bus_stops"][data["bus_stops"]["stop_id"] == sid]
                if len(row) > 0:
                    lat = row.iloc[0]["latitude"]
                    lon = row.iloc[0]["longitude"]
                    coords.append((lat, lon))
                    folium.CircleMarker(
                        location=[lat, lon], radius=4, color=color,
                        fill=True, fill_opacity=0.8,
                        popup=f"Route {ridx + 1}, Stop {sid}",
                    ).add_to(m4)

            if len(coords) >= 2:
                folium.PolyLine(coords, color=color, weight=3,
                                opacity=0.7, popup=f"Route {ridx + 1}").add_to(m4)

        st_folium(m4, width=1000, height=600, key="optimized_map")

        st.subheader("Route Summary")
        if "route_summary" in st.session_state:
            st.dataframe(st.session_state["route_summary"], use_container_width=True)
    else:
        st.info("Click 'Run Optimization' in the sidebar to generate optimized routes.")


# ============================================================
# Tab 4: Results
# ============================================================
with tab4:
    st.header("Optimization Results")

    if "opt_routes" in st.session_state:
        optimized_routes = st.session_state["opt_routes"]
        greedy_routes = st.session_state["greedy_routes"]

        evaluator = RouteEvaluator(
            data["bus_stops"], data["metro_stations"], data["population_grid"]
        )

        # Coverage analysis
        st.subheader("Coverage Analysis")
        opt_coverage = evaluator.coverage_metrics(optimized_routes)
        greedy_coverage = evaluator.coverage_metrics(greedy_routes)

        cov_col1, cov_col2, cov_col3 = st.columns(3)
        cov_col1.metric(
            "Population Coverage (Optimized)",
            f"{opt_coverage['population_coverage_pct']:.1f}%",
            delta=f"+{opt_coverage['population_coverage_pct'] - greedy_coverage['population_coverage_pct']:.1f}%",
        )
        cov_col2.metric("Routes Used", opt_coverage["total_routes"])
        cov_col3.metric("Stops Used", opt_coverage["total_stops_used"])

        # Transfer times
        st.subheader("Transfer Time Distribution")
        transfer_df = evaluator.transfer_time_distribution(optimized_routes)
        if len(transfer_df) > 0:
            fig_transfer = px.histogram(
                transfer_df, x="transfer_time_min", nbins=30,
                title="Bus-to-Metro Transfer Time Distribution",
                labels={"transfer_time_min": "Transfer Time (min)"},
                color_discrete_sequence=["#00CC96"],
            )
            fig_transfer.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_transfer, use_container_width=True)

        # Cost analysis
        st.subheader("Cost Analysis")
        cost = evaluator.cost_per_rider(optimized_routes, predictions)
        cost_col1, cost_col2, cost_col3 = st.columns(3)
        cost_col1.metric("Cost per Rider", f"{cost['cost_per_rider_sar']:.2f} SAR")
        cost_col2.metric("Daily Riders", f"{cost['estimated_daily_riders']:,.0f}")
        cost_col3.metric("Daily Cost", f"{cost['daily_operating_cost_sar']:,.0f} SAR")

        # Convergence plot
        st.subheader("GA Convergence")
        if "convergence" in st.session_state:
            conv_df = st.session_state["convergence"]
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=conv_df["generation"], y=conv_df["max_fitness"],
                mode="lines", name="Best Fitness", line=dict(color="#EF553B"),
            ))
            fig_conv.add_trace(go.Scatter(
                x=conv_df["generation"], y=conv_df["avg_fitness"],
                mode="lines", name="Avg Fitness", line=dict(color="#636EFA"),
            ))
            fig_conv.update_layout(
                title="Genetic Algorithm Convergence",
                xaxis_title="Generation", yaxis_title="Fitness",
                template="plotly_dark", height=350,
            )
            st.plotly_chart(fig_conv, use_container_width=True)

        # Before/After comparison
        st.subheader("Before vs After Optimization")
        comparison = evaluator.before_after_comparison(
            optimized_routes, greedy_routes, predictions
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        # Route quality
        st.subheader("Route Quality Scores")
        quality_df = evaluator.route_quality_scores(optimized_routes, predictions)
        if len(quality_df) > 0:
            fig_quality = px.bar(
                quality_df.sort_values("quality_score", ascending=False).head(20),
                x="route_id", y="quality_score",
                title="Top 20 Routes by Quality Score",
                color="quality_score", color_continuous_scale="RdYlGn",
                labels={"route_id": "Route ID", "quality_score": "Quality Score"},
            )
            fig_quality.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_quality, use_container_width=True)
    else:
        st.info("Click 'Run Optimization' in the sidebar to see results.")

