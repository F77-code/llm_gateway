# Atomic Redis sliding-window limiter

## Problem and scope

The original two non-transactional pipelines allow multiple callers to observe
the same free slot and subsequently exceed the quota. Replace cleanup, counting,
conditional insertion and expiry with one Redis Lua operation. Preserve the
`RateLimitResult` interface and usage accounting. Authentication and demo behavior
are separate PRs.

## Contract

- For nondecreasing supplied timestamps, the active window is
  `(now - window_seconds, now]`; entries at the left boundary expire. Production
  callers use wall-clock Unix seconds; `now` is retained for deterministic tests.
- Only admitted calls create entries. Each call has a UUID member, including
  calls at identical timestamps and across independent processes.
- `remaining` is the post-decision number of free slots, clamped at zero.
- `reset_ts` is the ceiling of the oldest active admission plus the window width,
  for both accepted and rejected calls. It identifies the next slot release;
  rounding must never suggest retry before a fractional boundary.
- TTL on admission remains `2 * window_seconds + 1` seconds. Rejection must not
  prolong it. Empty windows may be replaced atomically by a fresh admission.
- Positive integer limits/windows and finite timestamps are required. Limits
  must be at most `2**53 - 1` (exact Lua integer representation); windows must be
  at most `2**31 - 1` seconds (safe Redis expiry range). Derived cutoff/reset
  values must remain finite. Invalid arguments fail before mutating Redis.
- A single Redis key is used; scripts are atomic for that key. Wall-clock skew
  across gateway processes is an explicit remaining limitation, not a claim of
  globally ordered distributed time. Policy/limit changes are not implemented
  in this PR.

## Verification

Retain all existing tests, enabling Lua support in fakeredis. Add real Redis
tests for exact/fractional window boundaries, remaining/reset, identical scores,
key independence, TTL retention/expiry and missing Redis failure. Run concurrent
admission with separate Redis clients, then repeat across spawned processes.
Assert both admitted count and final Redis cardinality equal the quota.

Use a UUID namespace per test and delete only owned keys. Compose provides a
dedicated ephemeral Redis service without a published port. Integration tests
require its explicit URL and ping it; no hidden skips or FLUSHDB/FLUSHALL.

Demonstrate that the same concurrency invariant fails with the original method.
A deterministic barrier after the original read phase may coordinate the
otherwise timing-sensitive race while keeping real Redis reads/writes. Preserve
evidence in PR validation; do not keep a broken implementation in production.

## Reproduce the original regression

The baseline commit is `10960f4`. After building the checks image, extract the
original source with Git on the host, then mount it only into a throwaway test
container:

```bash
git show 10960f4:app/services/redis.py > /tmp/llm-gateway-redis-original.py
docker compose --profile checks run --rm \
  -v /tmp/llm-gateway-redis-original.py:/app/app/services/redis.py:ro \
  checks pytest -q tests/integration/test_atomic_limiter.py::test_concurrent_clients_never_exceed_quota
```

This run must fail: 32 admissions and 32 stored members exceed a quota of 7.
The unmodified checks image must pass the same test. The barrier coordinates
only the original post-read scheduling gap; no Redis operations are faked.
