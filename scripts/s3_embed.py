"""Copy src/longvid_ov/*.py into the EMBEDDED block of the Kaggle script (a Kaggle script kernel is one file).

Run after editing src/longvid_ov:  .venv/bin/python scripts/s3_embed.py   (tests fail if the copies are stale)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "kaggle" / "s3_tiling_bench" / "s3_tiling_bench.py"
FILES = ["tiled_decode.py", "metrics.py", "bench_worker.py"]
BEGIN, END = "# --- BEGIN EMBEDDED", "# --- END EMBEDDED ---"


def embedded_block():
    parts = []
    for name in FILES:
        src = (ROOT / "src" / "longvid_ov" / name).read_text()
        assert "'''" not in src and not src.endswith("\\"), f"{name} cannot be embedded in a raw ''' string"
        parts.append(f'    "{name}": r\'\'\'{src}\'\'\',\n')
    return "EMBEDDED = {\n" + "".join(parts) + "}\n"


def main():
    text = TARGET.read_text()
    head, rest = text.split(BEGIN, 1)
    marker, rest = rest.split("\n", 1)
    _, tail = rest.split(END, 1)
    TARGET.write_text(head + BEGIN + marker + "\n" + embedded_block() + END + tail)
    print(f"embedded {FILES} into {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
