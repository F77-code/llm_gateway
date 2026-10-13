# Delivery plan

The gateway remains one modular FastAPI service in this repository. Preserve all
provider adapters and direct model selection. Add infrastructure only when its
responsibility is implemented and tested. PRs may be stacked on their predecessor;
they remain drafts until separately approved. No automatic merging or force push.

## PR order and acceptance gates

1. **Atomic Redis limiter** — see [01-atomic-limiter.md](01-atomic-limiter.md).
   One atomic admission operation; real Redis concurrency across connections and
   processes; demonstrated failure on the original implementation; preserved unit
   coverage; locked, container-only test harness. Root development rules and this
   roadmap belong to this PR.
2. **Client authentication** — see [02-authentication.md](02-authentication.md).
   Registered client keys with distinct stable client IDs; fail-closed auth;
   `/me/usage`; secrets absent from URL, storage keys and responses.
3. **Reproducible demo and CI** — see [03-demo-and-ci.md](03-demo-and-ci.md).
   Clean-checkout `docker compose up --build` and later `docker compose up`;
   mock providers without paid credentials; full container quality profile,
   application smoke test and CI using isolated real Redis.
4. **Explainable routing** — see [04-routing.md](04-routing.md).
   `extract-default`, two initial candidates, capability catalog, versioned policy,
   pure deterministic planner and auditable RoutePlan; dry-run and execution share
   one policy snapshot; explicit rejection when hard constraints are unsatisfied.
5. **Streaming execution** — two adapter contracts, SSE, end-to-end deadlines,
   bounded admission/queues, disconnect cleanup, controlled retries, fallback
   before the first fragment only, capability-preserving fallback and breakers.
6. **Durable resource accounting** — PostgreSQL request/attempt/usage ledger,
   versioned prices in integer micro-USD or Decimal, spend reservations,
   idempotent records, restart and partial failure tests, pending reconciliation
   for unknown usage. No claim of a transaction spanning Redis and PostgreSQL or
   guaranteed billing cancellation on disconnect.
7. **Evals and telemetry** — synthetic support-ticket extraction with checked
   expectations, tuning/held-out split, economical/strong/policy comparison;
   schema validity, field accuracy, cost, TTFT and gateway overhead; OpenTelemetry
   stages/attempts and aggregate Prometheus metrics. Alternative quality requires
   separate eval runs; dry-run alone cannot establish it.
8. **TypeScript interface and E2E** — route reasons, attempts, usage, policy
   validation, version diff and rollback. Core user flows have E2E coverage.

Later work uses `providers`, `routing`, `execution`, `admission`, `usage` and
`telemetry` responsibility boundaries. New `frontend/`, `evals/`, `benchmarks/`
and `deploy/` stay in this repository. Learned routing, semantic caching, MCP
gateway and agent runtime remain future experiments. Avoid novelty/superiority
claims without measurements.

Each PR requires behavior checks, container commands, independent agent review
and a draft PR with validation evidence. Finish a gate before starting its
dependent implementation; report external access blockers precisely while
continuing independent work.
