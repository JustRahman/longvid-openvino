"""Refresh the patch and embed it (and the Python reference) into the step 5 Kaggle script (a script kernel is one file).

Run after changing the branch `ltx-temporal-vae-tiling` in third_party/openvino.genai or src/longvid_ov:
    python3 scripts/s5_embed_patch.py
It only runs git (rev-parse / format-patch) in the clone and rewrites two files:
    patches/0001-ltx-temporal-vae-tiling.patch          git format-patch of the branch against the cloned HEAD
    kaggle/s5_genai_build/s5_genai_build.py             the block between BEGIN GENERATED and END GENERATED
The patch is embedded as base64(zlib) because a patch must survive byte for byte (trailing spaces, tabs).
"""
import base64
import hashlib
import subprocess
import textwrap
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENAI = ROOT / "third_party" / "openvino.genai"
BRANCH, UPSTREAM = "ltx-temporal-vae-tiling", "origin/master"
PATCH = ROOT / "patches" / "0001-ltx-temporal-vae-tiling.patch"
TARGET = ROOT / "kaggle" / "s5_genai_build" / "s5_genai_build.py"
FILES = ["tiled_decode.py", "bench_worker.py"]
BEGIN, END = "# --- BEGIN GENERATED", "# --- END GENERATED ---"


def git(*args):
    return subprocess.run(["git", "-C", str(GENAI), *args], check=True, capture_output=True).stdout


def generated_block(base, patch):
    b64 = "\n".join(textwrap.wrap(base64.b64encode(zlib.compress(patch, 9)).decode(), 110))
    parts = []
    for name in FILES:
        src = (ROOT / "src" / "longvid_ov" / name).read_text()
        assert "'''" not in src and not src.endswith("\\"), f"{name} cannot be embedded in a raw ''' string"
        parts.append(f'    "{name}": r\'\'\'{src}\'\'\',\n')
    return (f'BASE_COMMIT = "{base}"\nPATCH_SHA256 = "{hashlib.sha256(patch).hexdigest()}"\n'
            f'PATCH_B64 = """\n{b64}\n"""\nEMBEDDED = {{\n{"".join(parts)}}}\n')


def main():
    base = git("merge-base", BRANCH, UPSTREAM).decode().strip()
    commits = int(git("rev-list", "--count", f"{base}..{BRANCH}").decode())
    assert commits == 1, f"{BRANCH} has {commits} commits on top of {base[:9]}: squash them into one"
    patch = git("format-patch", "-1", "--stdout", BRANCH)
    PATCH.parent.mkdir(exist_ok=True)
    PATCH.write_bytes(patch)
    text = TARGET.read_text()
    head, rest = text.split(BEGIN, 1)
    marker, rest = rest.split("\n", 1)
    _, tail = rest.split(END, 1)
    TARGET.write_text(head + BEGIN + marker + "\n" + generated_block(base, patch) + END + tail)
    print(f"base {base}\npatch {PATCH.relative_to(ROOT)}: {len(patch)} bytes, {patch.count(b'diff --git')} files\n"
          f"embedded patch + {FILES} into {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
