#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m src.data.merge_sources
python -m src.features.build_features
python -m src.models.train_random_forest
