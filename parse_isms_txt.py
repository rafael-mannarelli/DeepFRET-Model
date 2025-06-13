import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import lib.ml


def parse_isms_txt(txt_path, outdir):
    txt_path = Path(txt_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Find header start
    with txt_path.open("r") as f:
        lines = f.readlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("D-Dexc"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Header line starting with 'D-Dexc' not found")

    # Correction ici
    df = pd.read_csv(
        txt_path,
        skiprows=start_idx + 1,
        sep='\s+',
        names=["DD", "DA", "AA", "S", "E"],
    )

    X_vals = df[["DD", "DA", "AA"]].astype(np.float32).values
    n_frames = X_vals.shape[0]

    X = X_vals.reshape(1, n_frames, 3)

    name = txt_path.stem
    np.savez_compressed(outdir / f"X_{name}.npz", X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse intensity txt file")
    parser.add_argument("txt", help="Path to exported txt file")
    parser.add_argument("--outdir", "-o", default="./data", help="Output directory")
    args = parser.parse_args()

    parse_isms_txt(args.txt, args.outdir)
