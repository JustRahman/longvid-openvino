"""Copy src/longvid_ov/*.py into the EMBEDDED block of the two step 8 Kaggle scripts (a script kernel is one file).

Run after editing src/longvid_ov:  python3 scripts/s8_embed.py   (standard library only)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [ROOT / "kaggle" / "s8a_long_and_tuning" / "s8a_long_and_tuning.py",
           ROOT / "kaggle" / "s8b_memory_and_int8" / "s8b_memory_and_int8.py"]
FILES = ["tiled_decode.py", "metrics.py", "bench_worker.py", "s8_common.py", "s8_worker.py"]
BEGIN, END = "# --- BEGIN EMBEDDED", "# --- END EMBEDDED ---"


def embedded_block():
    parts = []
    for name in FILES:
        src = (ROOT / "src" / "longvid_ov" / name).read_text()
        assert "'''" not in src and not src.endswith("\\"), f"{name} cannot be embedded in a raw ''' string"
        parts.append(f'    "{name}": r\'\'\'{src}\'\'\',\n')
    return "EMBEDDED = {\n" + "".join(parts) + "}\n"


def main():
    for target in TARGETS:
        text = target.read_text()
        head, rest = text.split(BEGIN, 1)
        marker, rest = rest.split("\n", 1)
        _, tail = rest.split(END, 1)
        target.write_text(head + BEGIN + marker + "\n" + embedded_block() + END + tail)
        print(f"embedded {FILES} into {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
