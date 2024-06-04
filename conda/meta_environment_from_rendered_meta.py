from pathlib import Path

_dir = Path(__file__).resolve().parent


with open(_dir / "rendered-meta.yaml", mode="rt", encoding="utf-8") as f:
    dependencies = []
    # Collect all dependencies, from all environments
    in_requirements_section = False
    for num, line in enumerate(f):
        if line.strip() == "run:":
            # Run requirements are not included in the meta_environment.yaml,
            # since 'conda render' does not attempt to resolve them
            break
        if in_requirements_section and line.strip() not in ("build:", "host:"):
            if "::" not in line:
                dependencies.append(line)
        if line.strip() == "requirements:":
            in_requirements_section = True
    # De-duplicate
    dependencies = sorted(set(dependencies))
    dependencies.insert(0, "dependencies:\n")

with open(_dir / "meta_environment.yaml", mode="wt", encoding="utf-8") as f:
    f.writelines(dependencies)
