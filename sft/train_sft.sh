#!/bin/bash
# Full-parameter SFT with LlamaFactory on 4 GPUs (vision tower frozen).
#
# usage:
#   bash sft/train_sft.sh phyreason_method_8b        # one config from sft/configs/
#   bash sft/train_sft.sh phyreason_method_8b physics_baseline_4b ...
#
# Config names: {phyreason,physics}_{method,baseline}_{4b,8b,12b}   (12b = google/gemma-3-12b-it)
#   method   = Modeling-Sloving traces (input to rl/train_method.sh)
#   baseline = plain step-by-step solutions (input to rl/train_baseline.sh)
# Output: outputs/sft/<config name>/
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

[ $# -ge 1 ] || { sed -n '2,11p' "$0"; exit 1; }

export FORCE_TORCHRUN=1
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

for name in "$@"; do
    config="sft/configs/${name}.yaml"
    [ -f "${config}" ] || { echo "[sft] no such config: ${config}" >&2; exit 1; }
    echo "[sft] training ${name}"
    llamafactory-cli train "${config}"
    case "${name}" in
        *_12b)
            # LlamaFactory drops Gemma-3's extra_special_tokens from the saved tokenizer
            # config; without them vLLM cannot build the Gemma-3 processor.
            python3 - "outputs/sft/${name}/tokenizer_config.json" <<'PY'
import json, sys
path = sys.argv[1]
cfg = json.load(open(path))
if not isinstance(cfg.get("extra_special_tokens"), dict) or "image_token" not in cfg["extra_special_tokens"]:
    cfg["extra_special_tokens"] = {"image_token": "<image_soft_token>",
                                   "boi_token": "<start_of_image>", "eoi_token": "<end_of_image>"}
    json.dump(cfg, open(path, "w"), indent=2, ensure_ascii=False)
PY
            ;;
    esac
done
