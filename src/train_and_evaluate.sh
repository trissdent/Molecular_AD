#!/bin/bash
# Train one experiment, then extract latents and run post-analysis on that same run.
#   ./train_and_evaluate.sh configs/dti.yaml       (smri.yaml / dti.yaml / both.yaml ...)
# Safe to run several at once (e.g. one per GPU with CUDA_VISIBLE_DEVICES).

set -e

if [ -z "$1" ]; then
    echo "usage: ./train_and_evaluate.sh <override yaml>"
    exit 1
fi

EXP="$(realpath "$1")"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME_FILE="$(mktemp)"
trap 'rm -f "$RUN_NAME_FILE"' EXIT

cd "$PROJECT_ROOT/src"
python train_pipeline.py --exp "$EXP" --run-name-file "$RUN_NAME_FILE"

RUN_NAME="$(cat "$RUN_NAME_FILE")"
if [ -z "$RUN_NAME" ]; then
    echo "Training did not report its run name"
    exit 1
fi
export RUN_DIR="$PROJECT_ROOT/checkpoints/$RUN_NAME"

echo "Run name: $RUN_NAME"
echo "RUN_DIR: $RUN_DIR"

python extract_latents.py

cd "$PROJECT_ROOT"

python post_analysis/s1_latent_space.py
python post_analysis/s2_index.py
python post_analysis/s3_alignment.py
python post_analysis/s4_apoe.py

echo "Full pipeline completed: $RUN_NAME"