# physics_reasoning_neurips_2026

Code and data for the NeurIPS 2026 paper **"Axiomatic World Modeling for Physics Reasoning"**.
Author: Xinye Yang (XinyeYang2027@u.northwestern.edu)


## Contents

```
data/
  sft/          LlamaFactory SFT data (+ dataset_info.json)
                  {phyreason,physics}_method_{train,test}.json    Modeling-Sloving traces
                  {phyreason,physics}_baseline_{train,test}.json  plain step-by-step solutions
  images/       PhysReason problem images
  rl/           GRPO data (verl parquet): phyreason/, phyreason_baseline/, physics/
  checkers/     per-problem checkers for the checker reward
  rubrics/      per-problem rubrics for the rubric-as-reward baseline
  eval/         test splits: {phyreason,physics}_{method,baseline}_test.jsonl
sft/            SFT configs and launcher (LlamaFactory)
rl/             GRPO launchers (verl) and reward functions (rl/reward/)
eval/           evaluation (vLLM generation + LLM judge)
```

| dataset | train | test | source |
|---|---|---|---|
| PhysReason (`phyreason`) | 1,080 | 120 | [PhysReason](https://huggingface.co/datasets/zhibei1204/PhysReason) (MIT) |
| PHYSICS (`physics`) | 897 | 99 | [PHYSICS](https://github.com/yale-nlp/Physics) (MIT) |

## Setup

Install [LlamaFactory](https://github.com/hiyouga/LlamaFactory#installation) and
[verl](https://verl.readthedocs.io/en/latest/start/install.html) (with the vLLM backend)
following their official instructions. All commands below are run from the root of this
repository: `sft/` in the LlamaFactory environment, `rl/` and `eval/` in the verl environment.
In the verl environment, additionally run:

```bash
pip install -r requirements.txt       # reward / eval dependencies
cp .env.example .env                  # then fill in your keys
```

API keys are read from the environment (or `.env`):

| variable | used by |
|---|---|
| `OPENAI_API_KEY` | an [OpenRouter](https://openrouter.ai) key: evaluation judge, PMF reward, baseline rewards |
| `WANDB_API_KEY` | optional, enables wandb logging during RL |
| `HF_TOKEN` | optional, only needed for gated models |

Models (`--size` / config suffix): `4b` = [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct),
`8b` = [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
`12b` = [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it); pseudocode translator
[`Qwen/Qwen2.5-Coder-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct).

## Evaluate the released checkpoints

| checkpoint | trained on | base model |
|---|---|---|
| [`AsherYang/phyreason_rlawm_4b`](https://huggingface.co/AsherYang/phyreason_rlawm_4b) | PhysReason | Qwen/Qwen3-VL-4B-Instruct |
| [`AsherYang/phyreason_rlawm_8b`](https://huggingface.co/AsherYang/phyreason_rlawm_8b) | PhysReason | Qwen/Qwen3-VL-8B-Instruct |
| [`AsherYang/physics_rlawm_4b`](https://huggingface.co/AsherYang/physics_rlawm_4b) | PHYSICS | Qwen/Qwen3-VL-4B-Instruct |
| [`AsherYang/physics_rlawm_8b`](https://huggingface.co/AsherYang/physics_rlawm_8b) | PHYSICS | Qwen/Qwen3-VL-8B-Instruct |

The checkpoints are standard Hugging Face models and load directly in vLLM. Each command
below downloads the model, generates greedy answers for the test split on one GPU, and
scores them with the judge:

```bash
export OPENAI_API_KEY=<your OpenRouter key>

python eval/evaluate.py --model AsherYang/phyreason_rlawm_4b --dataset phyreason
python eval/evaluate.py --model AsherYang/phyreason_rlawm_8b --dataset phyreason
python eval/evaluate.py --model AsherYang/physics_rlawm_4b   --dataset physics
python eval/evaluate.py --model AsherYang/physics_rlawm_8b   --dataset physics
```

The script prints the overall **Subpass@1** and writes every response with its per-answer
judgement to `outputs/eval/<model>_<dataset>_<prompt>.json`.

**Options.**
- `--prompt method` (default) uses the Modeling-Sloving prompt the released checkpoints were
  trained with. Use `--prompt baseline` (plain step-by-step prompt) for base models and the
  baselines.


## Training

All training scripts take a model size: `4b`, `8b`, or `12b` (`google/gemma-3-12b-it`).

### 1. SFT

```bash
bash sft/train_sft.sh phyreason_method_8b       # Modeling-Sloving SFT (input to our RL)
bash sft/train_sft.sh phyreason_baseline_8b     # plain SFT (input to the baselines)
bash sft/train_sft.sh phyreason_method_12b      # Modeling-Sloving SFT on google/gemma-3-12b-it
```

Configs are `sft/configs/{phyreason,physics}_{method,baseline}_{4b,8b,12b}.yaml`. Models are written
to `outputs/sft/<config>/`.

### 2. RL: our method

```bash
bash rl/train_method.sh --dataset phyreason --size 8b
bash rl/train_method.sh --dataset physics   --size 8b
```

GPUs 0-2 run GRPO; GPU 3 serves the pseudocode translator with vLLM. PMF is scored by
`openai/gpt-4o-mini-2024-07-18` on OpenRouter. The script starts from
`outputs/sft/<dataset>_method_<size>` (override with `--policy`) and writes the final model
to `outputs/models/<dataset>_rlawm_<size>/`.


### 3. RL: baselines

```bash
bash rl/train_baseline.sh --kind rlvr       --dataset phyreason --size 8b
bash rl/train_baseline.sh --kind prm_rubric --dataset phyreason --size 8b
bash rl/train_baseline.sh --kind prm_only   --dataset phyreason --size 8b
bash rl/train_baseline.sh --kind prm_gt     --dataset phyreason --size 8b
```

| `--kind` | reward (judge: `openai/gpt-4o-mini-2024-07-18` on OpenRouter) |
|---|---|
| `rlvr` | fraction of ground-truth answers matched by the `[ans]` block |
| `prm_rubric` | rubric as reward: response graded against the problem's rubric (`data/rubrics/`) |
| `prm_only` | reference-free judgement of the reasoning |
| `prm_gt` | judgement given the reference answer and reference reasoning |

Baselines start from `outputs/sft/<dataset>_baseline_<size>` and train on all 4 GPUs. Evaluate
them with `--prompt baseline`:

```bash
python eval/evaluate.py --model outputs/models/baseline_phyreason_8b_rlvr --dataset phyreason --prompt baseline
```
