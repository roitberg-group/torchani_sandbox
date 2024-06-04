from pathlib import Path

_dir = Path(__file__).resolve().parent

# Filtered the rendered meta.yaml to make it readable by conda-build
with open(_dir / "rendered_meta.yaml", mode="rt", encoding="utf-8") as f:
    for num, line in enumerate(f):
        if line.strip() == "meta.yaml:":
            break
    next(f)
    filtered_lines = f.readlines()

with open(_dir / "rendered_meta.yaml", mode="wt", encoding="utf-8") as f:
    for num, line in enumerate(filtered_lines):
        if line.strip().startswith("version:"):
            break
    # Revert this change so GIT_DESCRIBE_TAG and GIT_BUILD_STR can be used
    # after the render
    filtered_lines[
        num
    ] = "  version: {{ environ.get('GIT_DESCRIBE_TAG', '2.3') }}.{{ environ.get('GIT_BUILD_STR', 'dev') }}\n"
    f.writelines(filtered_lines)

# Parse the rendered and filtered file to also create an environment file
with open(_dir / "rendered_meta.yaml", mode="rt", encoding="utf-8") as f:
    dependencies = []
    # Collect all dependencies, from all environments
    in_requirements_section = False
    for num, line in enumerate(f):
        if line.strip() == "run:":
            # Run requirements are not included in the meta_environment.yaml,
            # since 'conda render' does not attempt to resolve them
            break
        if in_requirements_section and line.strip() not in ("build:", "host:"):
            dependencies.append(line)
        if line.strip() == "requirements:":
            in_requirements_section = True
    # De-duplicate
    dependencies = sorted(set(dependencies))
    dependencies.insert(0, "dependencies:\n")

with open(_dir / "meta_environment.yaml", mode="wt", encoding="utf-8") as f:
    f.writelines(dependencies)
