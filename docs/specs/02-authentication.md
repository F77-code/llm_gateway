# Client authentication

Provide explicit client registrations containing a stable client ID and a hash
of a high-entropy bearer key. Keep provider credentials separate. Reject missing,
malformed, unknown and revoked keys; do not accept arbitrary tokens when unset.
Use client IDs for rate limits and usage, with tests for isolation and key
rotation. The subsequent infrastructure PR provides a clearly documented
demo-only registration; this PR defaults to an empty, fail-closed registry.

Replace `/stats/{api_key}` with authenticated `/me/usage`. Secrets must not occur
in URLs, logs, Redis keys or responses. Document the migration of existing
secret-indexed aggregates rather than silently implying history is transferred.
Test unauthorized access and cross-client isolation as well as valid usage.

`GATEWAY_CLIENTS` is a JSON list of frozen registrations with public, namespace-safe
IDs and SHA-256 credential hashes. A boolean `enabled` supports revocation; two
distinct hashes may share an ID for rotation. Reject globally duplicated hashes
and hashes of any configured client ID. Generate 32 random bytes per credential
using `scripts/create_client_key.py` in Docker; configuration stores only hashes.
Configuration takes effect on restart. Duplicate Authorization headers are
rejected. Valid Bearer syntax permits one or more literal spaces after the scheme.

Existing secret-indexed Redis usage is not automatically migrated. New records
use client IDs; historic aggregates can be exported privately before upgrading.
Float cost aggregates remain a documented interim format until durable usage.
