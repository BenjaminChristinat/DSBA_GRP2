# Thesis Utilities

Small helpers that were originally bundled with the thesis submission and still serve as useful references when adjusting collectors.

- `build_city_rectangles.py` – early rectangle builder that experimented with different padding strategies.
- `fetch_bounds.py` – quick helper to fetch bounding boxes for an ad-hoc list of cities.
- `fast_get_restaurants.py` – prototype Google Places crawler (single-threaded) kept for provenance.
- `failed_run*.csv` – snapshots captured while debugging extraction failures.
- `cities_zones.rtf` – notes describing the manual zoning decisions.

Feel free to lift logic from these scripts into the production collectors or rerun them directly via the repo `.venv`.
