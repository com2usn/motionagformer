#!/usr/bin/env bash
# Evaluate MotionAGFormer-L on MPI-INF-3DHP (prints MPJPE).
# Run from anywhere:  source scripts/evaluate-3dhp.sh   (or: bash scripts/evaluate-3dhp.sh)
# Pick another GPU:   CUDA_VISIBLE_DEVICES=1 source scripts/evaluate-3dhp.sh
# Runs in a subshell, so sourcing it does not change your cwd/venv and errors don't close your shell.
(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    source .venv/bin/activate || exit 1
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

    for f in checkpoint/motionagformer-l-mpi.pth.tr data/motion3d/data_train_3dhp.npz data/motion3d/data_test_3dhp.npz; do
        if [ ! -f "$f" ]; then
            echo "[ERROR] Missing $f (see README 'MPI-INF-3DHP' and 'Evaluation')." >&2
            exit 1
        fi
    done

    python train_3dhp.py --eval-only --checkpoint checkpoint --checkpoint-file motionagformer-l-mpi.pth.tr \
        --config configs/mpi/MotionAGFormer-large.yaml
)
