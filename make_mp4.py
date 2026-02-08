import argparse
import os
import re
import subprocess
from pathlib import Path


def extract_step(name: str) -> int:
    m = re.search(r"_(\d+)\.png$", name)
    return int(m.group(1)) if m else 10**12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path("output")
    if not out_dir.is_dir():
        raise SystemExit("Brak katalogu output")

    pngs = [p for p in out_dir.glob("*.png") if "FINAL" not in p.name.upper()]
    if not pngs:
        raise SystemExit("Brak plików PNG")

    m = re.match(r"(sc\d+)_", pngs[0].name)
    prefix = m.group(1) if m else "animation"

    pngs = sorted(pngs, key=lambda p: extract_step(p.name))

    list_path = out_dir / "_frames.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in pngs:
            f.write(f"file '{p.name}'\n")

    out_mp4 = Path(f"{prefix}.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-r", str(args.fps),
        "-f", "concat", "-safe", "0",
        "-i", "_frames.txt",
        "-vf", "format=yuv420p",
        "-c:v", "libx264",
        "-crf", "18",
        str(out_mp4),
    ]

    subprocess.run(cmd, check=True, cwd=str(out_dir))
    os.remove(list_path)
    print(f"[OK] {out_mp4}")


if __name__ == "__main__":
    main()
