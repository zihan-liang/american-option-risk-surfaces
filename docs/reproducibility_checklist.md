# Reproducibility Checklist

- [ ] Record Python version and package versions.
- [ ] Run `python -m unittest discover -s tests`.
- [ ] Run `python -m compileall -q src tests experiments`.
- [ ] Store each experiment config beside outputs or in manifests.
- [ ] Record random seed for datasets and surrogate training.
- [ ] Keep generated `.npz`, `.pt`, plots, and logs out of Git unless explicitly curated.
- [ ] Link every paper figure to a script, config, and source data file.
- [ ] Re-run solver validation after changing numerical settings.
