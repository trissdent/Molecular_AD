#!/bin/bash

set -e

PROJECT_ROOT="/home/minhtri/Molecular_AD"

BEFORE_RUN=$(ls -td "$PROJECT_ROOT"/logs/*/ 2>/dev/null | head -1 || true)

cd "$PROJECT_ROOT/src"
python train_pipeline.py

cd "$PROJECT_ROOT"

LATEST_RUN=$(ls -td "$PROJECT_ROOT"/logs/*/ | head -1)

if [ "$LATEST_RUN" = "$BEFORE_RUN" ]; then
    echo "No new training run folder was created"
    exit 1
fi

RUN_NAME=$(basename "$LATEST_RUN")
export RUN_DIR="$PROJECT_ROOT/checkpoints/$RUN_NAME"

echo "Run name: $RUN_NAME"
echo "RUN_DIR: $RUN_DIR"

python post_analysis/extract_latents.py
python post_analysis/analyze_dci.py
python post_analysis/compute_index.py
python post_analysis/validate_index.py
python post_analysis/plot_latent_map.py

echo "Full pipeline completed"