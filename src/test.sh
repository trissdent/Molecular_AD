#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_NAME="2026-08-16_13-48-33"
export RUN_DIR="$PROJECT_ROOT/checkpoints/$RUN_NAME"

cd "$PROJECT_ROOT/src" && python test_pipeline.py

cd "$PROJECT_ROOT"

python post_analysis/s1_latent_space.py
python post_analysis/s2_index.py
python post_analysis/s3_alignment.py
python post_analysis/s4_apoe.py