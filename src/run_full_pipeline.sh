#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BEFORE_RUN=$(ls -td "$PROJECT_ROOT"/logs/*/ 2>/dev/null | head -1 || true)

cd "$PROJECT_ROOT/src"
python train_pipeline.py

LATEST_RUN=$(ls -td "$PROJECT_ROOT"/logs/*/ | head -1)

if [ "$LATEST_RUN" = "$BEFORE_RUN" ]; then
    echo "No new training run folder was created"
    exit 1
fi

RUN_NAME=$(basename "$LATEST_RUN")
export RUN_DIR="$PROJECT_ROOT/checkpoints/$RUN_NAME"

echo "Run name: $RUN_NAME"
echo "RUN_DIR: $RUN_DIR"

python test_pipeline.py

cd "$PROJECT_ROOT"

python post_analysis/s1_latent_space.py
python post_analysis/s2_index.py
python post_analysis/s3_alignment.py

echo "Full pipeline completed"