# Transparency-First CNN Checklist

Goal: prioritize transparent decision-making over raw accuracy.

Use this as your manual progress tracker while implementation continues.

## Phase 0 — Scope And Audit Setup
- [x] Confirm project thesis sentence in README and report deck.
- [x] Finalize one audit output root: `outputs/fusion/audit/<timestamp>/`.
- [x] Finalize one master predictions CSV schema used across scripts.
- [x] Define "done" criteria for transparency deliverables (not just accuracy).

## Phase 1 — Organized Probabilities
- [x] Add `prob_top1`, `prob_top2`, `margin_top1_top2` to prediction CSVs.
- [x] Add optional uncertainty proxy (`entropy`).
- [x] Flag high-confidence errors in CSV (`high_conf_wrong`).
- [x] Ensure class probability columns are stable (`prob_<class_slug>`).

## Phase 2 — Grad-CAM As Core Attribution
- [x] Keep predicted-class Grad-CAM outputs in training/eval flow.
- [x] Keep true-class Grad-CAM option for error analysis.
- [x] Keep shallow vs deep RGB comparison panel.
- [x] Add short "how to read Grad-CAM" note in README/report.

## Phase 3 — Occlusion / Patch Masking
- [x] Implement occlusion map generation (single-image).
- [x] Add batch runner for top-N validation samples.
- [x] Save occlusion maps to audit folder with consistent naming.
- [x] Add CSV summary of occlusion sensitivity scores.

## Phase 4 — Integrated Gradients
- [x] Implement IG for RGB input path.
- [x] Decide and document baseline choice (black / gray / mean_color of input).
- [x] Add optional IG export in eval scripts.
- [x] Compare IG vs Grad-CAM on same samples.

## Phase 5 — Embeddings and K-NN Neighbors
- [x] Choose embedding source layer for nearest-neighbor lookup.
- [x] Export train embeddings once per model version.
- [x] Generate top-k nearest training examples per evaluated sample.
- [x] Save neighbor table and sample grids into audit outputs.

## Phase 6 — Counterfactuals (Practical)
- [x] Implement contrastive examples (nearest different class).
- [x] Implement minimal masking-based flip attempt (bounded search).
- [x] Log what change flipped/not flipped prediction.
- [x] Add caveat text: counterfactuals are heuristic, not causal proof.

## Phase 7 — Structured Reporting
- [x] Standardize per-run folders: metrics / attribution / neighbors / failures.
- [x] Export per-class metrics and confusion matrix tables.
- [x] Export slice metrics (by area/source if metadata available).
- [x] Build failure gallery (high-confidence wrong, low-margin confusing cases).

## Data And Shortcut Audits
- [x] Re-check background bias with current Grad-CAM outputs.
- [x] Re-run with `--augment --anti-background-aug` and compare.
- [x] Document if background focus decreased or persisted.
- [x] Record remaining failure patterns and mitigation ideas.

## Reproducibility And Ops
- [ ] Ensure `.gitignore` excludes outputs/caches as intended.
- [ ] Confirm `requirements.txt` installs full pipeline.
- [ ] Add exact run commands used for final report.
- [ ] Pin final model artifact names referenced in report.

## Final Sign-Off
- [ ] Transparency outputs are complete and interpretable.
- [ ] Limitations section is explicit and honest.
- [ ] Final README workflow matches actual scripts/flags.
- [ ] Repo state clean and ready for commit/push.

