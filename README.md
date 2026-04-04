# Riyadh Bus Network Optimizer

Optimizes feeder bus routes connecting Riyadh neighborhoods to the 85-station metro network. Uses XGBoost for demand prediction and DEAP genetic algorithms for route optimization, targeting max population coverage with min travel time.

## Install

```bash
pip install -r requirements.txt
```

## Run

```python
from src.data_generator import RiyadhDataGenerator
from src.demand_model import DemandPredictor
from src.route_optimizer import RouteOptimizer

data = RiyadhDataGenerator().generate_all()
predictor = DemandPredictor()
features = predictor.prepare_features(data["bus_stops"], data["metro_stations"], data["population_grid"])
demand = predictor.generate_demand_labels(features)
predictor.train(features, demand)

optimizer = RouteOptimizer(max_routes=45, fleet_size=90)
optimizer.setup(data["bus_stops"], data["metro_stations"], data["population_grid"], predictor.predict(features))
routes = optimizer.optimize(n_generations=100)
```

```bash
# or launch the dashboard
streamlit run app.py
```

## Project Structure

```
src/
    data_generator.py     Synthetic stop, station, and population data
    demand_model.py       XGBoost demand prediction
    route_optimizer.py    DEAP genetic algorithm route optimization
    network_graph.py      Graph representation of bus network
    evaluation.py         Coverage and travel time metrics
app.py                    Streamlit dashboard
```

## License

MIT License
