# inverse-clipasso

Inverse CLIPasso prototype that refines user sketches toward a target semantic label.

## Structure

- `env.yml` / `requirements.txt` – dependency manifests for the Python environment.
- `src/` – core library code (data, rendering, CLIP, optimization, evaluation, utilities).
- `experiments/` – YAML configs describing optimization runs.
- `notebooks/` – exploratory research notebooks.
- `outputs/` – run artifacts (PNGs, SVGs, metrics JSON).

## Getting Started

1. Create the environment:
   ```bash
   conda env create -f env.yml    # or python -m venv && pip install -r requirements.txt
   conda activate inverse-clipasso
   ```
2. Launch experiments via your preferred runner, pointing to a config in `experiments/configs`.
3. Render or inspect results in `notebooks/` using utilities under `src/utils`.

## Notes

- Source files are intentionally skeletal; fill in logic as the project matures.
- Generated outputs should stay inside `outputs/` to keep the repo clean.
