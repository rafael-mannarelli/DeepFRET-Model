import argparse
from pathlib import Path

from parse_isms_txt import parse_isms_txt


def parse_isms_dir(directory, outdir):
    directory = Path(directory)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for txt_file in sorted(directory.glob("*.txt")):
        parse_isms_txt(txt_file, outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse all txt traces in a directory using parse_isms_txt"
    )
    parser.add_argument("directory", help="Directory containing txt files")
    parser.add_argument(
        "--outdir",
        "-o",
        default="./data",
        help="Directory where parsed npz files will be stored",
    )
    args = parser.parse_args()
    parse_isms_dir(args.directory, args.outdir)
