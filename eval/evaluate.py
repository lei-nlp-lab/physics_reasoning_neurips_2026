"""Evaluate a model on the PhysReason (120) or PHYSICS (99) test split.

Generation: vLLM, greedy decoding. Scoring: an LLM judge (eval/judge.py) matches the
ground-truth sub-answers against the final [ans] block of each response.

usage:
    export OPENAI_API_KEY=<your OpenRouter key>
    python eval/evaluate.py --model AsherYang/phyreason_rlawm_8b --dataset phyreason
    python eval/evaluate.py --model outputs/models/baseline_physics_8b_rlvr --dataset physics --prompt baseline

--prompt method   : Modeling-Sloving prompt (our SFT/RL models, incl. the released checkpoints)
--prompt baseline : plain step-by-step prompt (base models and the baselines)
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import configure_judge, evaluate_sample_response  # noqa: E402

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF repo id or local model directory")
    p.add_argument("--dataset", required=True, choices=["phyreason", "physics"])
    p.add_argument("--prompt", default="method", choices=["method", "baseline"])
    p.add_argument("--output", default=None, help="default: outputs/eval/<model>_<dataset>_<prompt>.json")
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--judge_model", default=os.environ.get("JUDGE_MODEL", "deepseek/deepseek-v4-flash"))
    p.add_argument("--judge_base_url", default=os.environ.get("JUDGE_BASE_URL", "https://openrouter.ai/api/v1"))
    p.add_argument("--judge_concurrency", type=int, default=64)
    return p.parse_args()


def load_items(dataset, prompt):
    path = REPO / "data" / "eval" / f"{dataset}_{prompt}_test.jsonl"
    items = [json.loads(line) for line in open(path, encoding="utf-8")]
    for it in items:
        it["image_path"] = str(REPO / it["image"]) if it.get("image") else None
    return items


def is_vl_model(model):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True).to_dict()
    arch = (cfg.get("architectures") or [""])[0].lower()
    return "vision" in arch or "vl" in cfg.get("model_type", "").lower() or "conditional" in arch


def generate(model, items, max_new_tokens, tp, gpu_mem):
    from vllm import LLM, SamplingParams

    sampling = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
    llm = LLM(model=model, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem,
              trust_remote_code=True, max_model_len=14336)
    if is_vl_model(model):
        from PIL import Image
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        inputs = []
        for it in items:
            msgs, img = [], it["image_path"]
            for m in it["messages"]:
                if m["role"] == "user" and img:
                    # The image goes into the first user turn, before its text.
                    msgs.append({"role": "user", "content": [{"type": "image", "image": img},
                                                             {"type": "text", "text": m["content"]}]})
                else:
                    msgs.append(m)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if it["image_path"]:
                image = Image.open(it["image_path"]).convert("RGB")
                inputs.append({"prompt": text, "multi_modal_data": {"image": image}})
            else:
                inputs.append({"prompt": text})
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        inputs = [tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True)
                  for it in items]
    outputs = llm.generate(inputs, sampling_params=sampling)
    return [o.outputs[0].text for o in outputs]


def subpass_at_1(results):
    """Mean over problems of the fraction of sub-answers judged correct (skipped judge calls excluded)."""
    scores = [r["score"] for r in results if r["score"] is not None]
    return sum(scores) / len(scores) if scores else 0.0, len(scores)


def main():
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY (an OpenRouter key) for the judge.")
    configure_judge(args.judge_model, args.judge_base_url, api_key)

    items = load_items(args.dataset, args.prompt)
    print(f"[eval] {len(items)} problems ({args.dataset}, {args.prompt} prompt), "
          f"{sum(bool(i['image_path']) for i in items)} with images")
    responses = generate(args.model, items, args.max_new_tokens, args.tp, args.gpu_memory_utilization)

    def judge(pair):
        it, resp = pair
        return evaluate_sample_response(it["answer"], resp)

    with ThreadPoolExecutor(args.judge_concurrency) as ex:
        judged = list(tqdm(ex.map(judge, zip(items, responses)), total=len(items), desc="judge"))

    results = []
    for it, j in zip(items, judged):
        results.append({
            "prob_id": it["prob_id"], "difficulty": it["difficulty"], "subdomain": it["subdomain"],
            "answer": it["answer"], "response": j["response"],
            "predicted_answers": j["predicted_answers"], "has_answer_tag": j["has_answer_tag"],
            "score": j["evaluation"]["score"], "sub_results": j["evaluation"]["sub_results"],
        })
    score, judged = subpass_at_1(results)

    out = Path(args.output or REPO / "outputs" / "eval" /
               f"{Path(args.model.rstrip('/')).name}_{args.dataset}_{args.prompt}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "dataset": args.dataset, "prompt": args.prompt,
               "judge_model": args.judge_model, "Subpass@1": score, "judged": judged,
               "results": results},
              open(out, "w"), ensure_ascii=False, indent=1)

    note = "" if judged == len(results) else f" ({judged}/{len(results)} judged)"
    print(f"\n{args.model} | {args.dataset} | Subpass@1 = {score:.4f}{note}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
