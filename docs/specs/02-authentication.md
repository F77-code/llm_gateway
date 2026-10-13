# Client authentication

Provide explicit client registrations containing a stable client ID and a hash
of a high-entropy bearer key. Keep provider credentials separate. Reject missing,
malformed, unknown and revoked keys; do not accept arbitrary tokens when unset.
Use client IDs for rate limits and usage, with tests for isolation and key
rotation. Keep a clearly documented demo-only registration.

Replace `/stats/{api_key}` with authenticated `/me/usage`. Secrets must not occur
in URLs, logs, Redis keys or responses. Document the migration of existing
secret-indexed aggregates rather than silently implying history is transferred.
Test unauthorized access and cross-client isolation as well as valid usage.
