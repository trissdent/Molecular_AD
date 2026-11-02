#!/bin/bash
# Re-run testing and post-analysis for an existing run.
#   ./test.sh <run_name>            e.g. ./test.sh 2026-11-02_10-00-00

set -e

if [ -z "$1" ]; then
    echo "usage: ./test.sh <run_name>"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_NAME="$1"
export RUN_DIR="$PROJECT_ROOT/checkpoints/$RUN_NAME"

if [ ! -d "$RUN_DIR" ]; then
    echo "No run folder: $RUN_DIR"
    exit 1
fi

cd "$PROJECT_ROOT/src" && python extract_latents.py

cd "$PROJECT_ROOT"

python post_analysis/s1_latent_space.py
python post_analysis/s2_index.py
python post_analysis/s3_alignment.py
python post_analysis/s4_apoe.py