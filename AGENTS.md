# Project rules

- Use official documentation, original research and the pinned HM implementation.
- Keep the KG project and historical custom controller out of this repository.
- Keep the six local modules focused; reuse upstream planning and perception.
- Prefer Miniforge, Conda and mamba. Preserve the working GPU container.
- Do not treat shorter code as simpler if it hides state, missing data or failures.
- Document scientific definitions and validation boundaries in docstrings and docs.
- Keep templates, model pins, seeds, source hashes, requests and failed attempts.
- Run CPU checks with `python -m pytest -q`, Ruff if installed, and `bash -n job.sbatch`.
- Never submit a paid job or silently change the research design as part of a cleanup.
