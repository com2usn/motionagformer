#!/usr/bin/env bash
# Evaluate MotionAGFormer-L on Human3.6M.
# Run from anywhere:  source scripts/evaluate-h36m.sh   (or: bash scripts/evaluate-h36m.sh)
# Pick another GPU:   CUDA_VISIBLE_DEVICES=1 source scripts/evaluate-h36m.sh
# Runs in a subshell, so sourcing it does not change your cwd/venv and errors don't close your shell.
(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    source .venv/bin/activate || exit 1
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

    for f in checkpoint/motionagformer-l-h36m.pth.tr data/motion3d/h36m_sh_conf_cam_source_final.pkl; do
        if [ ! -f "$f" ]; then
            echo "[ERROR] Missing $f (see README 'Dataset' and 'Evaluation')." >&2
            exit 1
        fi
    done
    if [ ! -d data/motion3d/H36M-243/train ] || [ ! -d data/motion3d/H36M-243/test ]; then
        echo "[INFO] Slicing H3.6M into 243-frame clips (first run only)..."
        (cd data/preprocess && python h36m.py --n-frames 243) || exit 1
    fi

    python train_h36m.py --eval-only --checkpoint checkpoint --checkpoint-file motionagformer-l-h36m.pth.tr \
        --config configs/h36m/MotionAGFormer-large.yaml
)
