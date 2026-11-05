# SwingSync project instructions

Read autonomy/CURRENT_STATUS.md and autonomy/backlog.json at every execution. Inspect repository and running jobs before selecting an unblocked increment. Complete working increments, verify them, obtain independent review for meaningful changes, and record evidence.

Coordinator owns architecture, shared interfaces, integration, and ALL Git operations. Use only this dedicated repository https://github.com/ryaaners/SwingSync on agent/swingsync; never force push or merge main automatically. At most four native agents including coordinator. Assign disjoint ownership. Roles live in autonomy/roles/ and project-scoped .codex/agents/; inherit model and permissions.

Never fabricate model results, trained checkpoints, advice validation, or résumé claims. Synthetic clips prove software behavior only. A missing checkpoint is unavailable, not an invitation to invent event predictions. No paid APIs, cloud inference, or LLM in product. Source videos, datasets, caches and checkpoints stay out of Git.

Preserve actual presentation timestamps. Never split random frames. Preserve GolfDB outer splits; group source recordings internally. CaddieSet outcomes and derivatives are forbidden predictors; group golfers and duplicate shots. Outcome integration requires feature equivalence and generalization gates. Advice must cite measurement, interval, evidence, reference and validated reliability; otherwise suppress it.

Run one expensive training job at a time. Record identity, fingerprint, seed, environment, config and resumable artifacts before launch. Never equate launched with completed. Record blockers once and work independently. No schedules without separate request.

Checks: `uv sync --locked`; `uv run pytest`; `uv run ruff check backend tests`; `cd frontend && npm ci && npm run build`. Update status/backlog after checks; inspect complete diff and stage intended files only. Fetch/check divergence, commit, push without force and verify remote SHA. Do not overwrite user changes.
