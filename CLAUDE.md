# riyadh-bus-network-optimizer

Bus route optimization for Riyadh public transport network.

**Setup:** `python -m venv .venv && source .venv/Scripts/activate && pip install -r requirements.txt`

**Test:** `pytest tests/ -v`

**Run:** `python -m src.main`

Core files: `src/route_optimizer.py` (includes `great_circle_km`), `src/demand_model.py`. Sparse docstrings. Casual commits.
