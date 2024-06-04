from pathlib import Path

_dir = Path(__file__).resolve().parent


with open(_dir / "meta.yaml", mode="rt", encoding="utf-8") as f:
    for num, line in enumerate(f):
        if line.strip() == "meta.yaml:":
            break
    next(f)
    filtered_file = f.read()


with open((_dir / "recipe") / "meta.yaml", mode="wt", encoding="utf-8") as f:
    f.write(filtered_file)
