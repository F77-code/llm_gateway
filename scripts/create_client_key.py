"""Generate a high-entropy bearer key and its registration inside Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.client_id):
        parser.error(
            "client ID must contain 1-64 letters, digits, underscores or hyphens"
        )
    token = secrets.token_urlsafe(32)
    print(
        json.dumps(
            {
                "api_key": token,
                "registration": {
                    "client_id": args.client_id,
                    "key_sha256": hashlib.sha256(token.encode()).hexdigest(),
                    "enabled": True,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
