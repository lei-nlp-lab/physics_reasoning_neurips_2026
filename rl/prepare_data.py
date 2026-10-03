"""Make a copy of an RL split whose image paths are absolute.

The released parquet files store PhysReason image paths relative to the repo root
(``data/images/PhysReason_img/<id>.jpg``). verl loads images inside Ray workers whose
working directory is not the repo root, so the training scripts call this once to
write ``outputs/rl_data/<name>/{train,test}.parquet`` with ``file://`` absolute paths.

usage: python rl/prepare_data.py <name>      # phyreason | phyreason_baseline | physics
"""

import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def main():
    name = sys.argv[1]
    src = REPO / "data" / "rl" / name
    dst = REPO / "outputs" / "rl_data" / name
    dst.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        df = pd.read_parquet(src / f"{split}.parquet")
        images = []
        for row in df["images"]:
            out = []
            for item in (list(row) if row is not None else []):
                path = REPO / item["image"]
                if not path.is_file():
                    raise FileNotFoundError(path)
                out.append({"image": f"file://{path}"})
            images.append(out)
        df["images"] = images
        df.to_parquet(dst / f"{split}.parquet", index=False)
        print(f"[prepare_data] {name}/{split}: {len(df)} rows -> {dst / f'{split}.parquet'}")


if __name__ == "__main__":
    os.chdir(REPO)
    main()
