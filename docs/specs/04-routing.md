# Explainable routing

Implement the first task profile `extract-default` with two initial candidates.
Catalog capabilities per concrete provider/model/adapter: JSON schema, tools,
streaming and context. Hard constraints are never relaxed silently.

Version policy snapshots. A pure deterministic planner filters candidates and
returns provider, upstream_model, policy version and exclusion reasons for every
alternative. If no candidate qualifies, return an explicit structured refusal.
Retain direct selection of existing models.

Dry-run performs no provider HTTP call. Dry-run and real execution invoke the
same algorithm against the same immutable snapshot. Tests cover deterministic
results, impossible constraints, unsupported adapter capabilities, policy
validation, version selection and preservation of direct routing.
