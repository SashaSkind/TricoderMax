# Data license notice

This is a **research prototype for prioritization, not a diagnostic device.**

The datasets it is developed and evaluated against are licensed for
**non-commercial use only**:

- **RSNA 2019 Intracranial Hemorrhage Detection** — RSNA challenge terms,
  non-commercial research use only.
- **CQ500** (qure.ai) — CC BY-NC-SA 4.0, non-commercial research use only.

Consequences, enforced by this repo:

- No imaging data or derived pixel artifacts (overlays, windowed slices, numpy
  volumes) are committed. `data/` and `artifacts/` are gitignored.
- This notice is printed on startup of the UI and the eval scripts.
- Any pretrained model weights reused here inherit the non-commercial terms of
  the data they were trained on.

Do not deploy this system clinically or commercially.
