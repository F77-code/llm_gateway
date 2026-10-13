# Development rules

- Preserve this repository, its history, existing provider adapters and user edits.
- Read the applicable stage specification in `docs/specs/` before changing behavior.
- Run dependency installation, application, lint, typing and tests only in Docker.
- Use uv and commit `uv.lock`; select dependency releases available by 2026-04-07.
- Keep Redis integration tests isolated from application Redis. Use namespaced keys,
  clean only owned keys, and fail when required Redis is unavailable.
- Test invariants, errors and concurrency for material behavior changes.
- Delegate implementation, testing and independent review to separate agents with
  explicit file ownership. Agents must not overwrite each other's edits.
- Use feature/fix branches and small Conventional Commits with real timestamps.
  Push to existing origin and open draft PRs; do not merge or force-push.
- Commit useful team documentation; keep temporary agent logs/reports out of Git.
- Report actual checks, limitations, PR links and exact Docker commands per stage.
