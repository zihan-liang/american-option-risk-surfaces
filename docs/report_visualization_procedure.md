# Report Visualization Procedure

This workflow defines what the manager thread should do when the user says:

> do the visualization of report

## Roles

- Manager thread: decides scope, delegates work, checks completion, and reports status to the user.
- Image drawer thread: studies reference-paper figures and designs professional code-generated visualizations.
- Report writer thread: incorporates the approved visualizations into the LaTeX report and recompiles the PDF.

Current dedicated threads:

- Image drawer: `019f31b1-048f-7e22-be19-9f103ba8d21e`
- Report writer: `019f274b-3c7b-78a0-908e-902e28a4010b`

## Default Procedure

1. Confirm the target report and reference paper paths.
2. Send the image drawer thread the reference paper and report context.
3. Ask image drawer to inspect the paper figures, identify the scientific purpose of each important visualization, and map useful figure ideas to this project.
4. Require a structured visualization brief with:
   - reference-paper figure idea,
   - scientific meaning,
   - project analogy,
   - required repo data/source files,
   - recommended output filename/path,
   - suggested caption,
   - priority: must-have, useful, or optional.
5. Ask image drawer to generate prototype code-based figures only when the figure is clearly useful.
6. Forward the visualization brief and any prototype figure paths to report writer.
7. Ask report writer to refine the LaTeX report and PDF using the brief.
8. Require report writer to verify the compiled PDF and report branch, commit, changed files, figures, and verification.
9. Manager reviews the completion reports and gives the user an English-first, Chinese-following summary.

## Claim Discipline

- CN/PSOR remains the benchmark and label generator.
- Do not claim PINN replaces PSOR.
- The supported Stage 23 claim is that `terminal_pde_weight_5` improves over the matched supervised neural baseline on the capped put campaign, including price, masked residual, boundary, and Delta diagnostics.

## Style Discipline

- Figures should be professional, simple, and report-ready.
- Prefer reproducible code-generated figures: Python/matplotlib, LaTeX/TikZ/PGFPlots, Mermaid, SVG, or similar.
- Save editable source plus rendered output whenever practical.
- Avoid decorative or fancy visuals unless scientifically necessary.
