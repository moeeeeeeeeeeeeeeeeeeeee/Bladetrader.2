# Data directory

- **Not committed:** local SQLite DB (`finhack.db` or path from `DATABASE_URL`), uploaded files, and generated snapshots or experiment outputs. See root `.gitignore`.

- **`case4_dataset_snapshot.jsonl`** (optional): produced by `scripts/build_case4_dataset_snapshot.py` for offline work (e.g. `scripts/validate_case4_earnings.py --offline-only`).

Regenerate local artifacts as needed; the repository stays small for browsing on GitHub.
