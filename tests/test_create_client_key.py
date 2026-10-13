"""Exercise the operator key-generation CLI inside the Docker checks image."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.auth import ClientRegistration
from tests.conftest import CLIENT_ID, GATEWAY_KEY, key_digest


def _run_generator(client_id: str) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_client_key.py"
    return subprocess.run(
        [sys.executable, str(script), "--client-id", client_id],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_generated_keys_are_independent_random_credentials_with_matching_hashes():
    generated = []
    for _ in range(2):
        result = _run_generator(CLIENT_ID)
        assert result.returncode == 0
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        assert set(payload) == {"api_key", "registration"}
        token = payload["api_key"]
        registration = ClientRegistration.model_validate(payload["registration"])
        assert len(token) >= 43
        assert len(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))) == 32
        assert registration.client_id == CLIENT_ID
        assert registration.enabled is True
        assert registration.key_sha256 == hashlib.sha256(token.encode()).hexdigest()
        assert token != CLIENT_ID
        generated.append((token, registration.key_sha256))
    assert generated[0][0] != generated[1][0]
    assert generated[0][1] != generated[1][1]


@pytest.mark.parametrize(
    "client_id", ["", "has:colon", "a" * 65, f"unsafe:{GATEWAY_KEY}"]
)
def test_invalid_client_id_exits_without_generating_or_echoing_secret(client_id):
    result = _run_generator(client_id)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "client ID must contain" in result.stderr
    assert GATEWAY_KEY not in result.stderr
    assert key_digest(GATEWAY_KEY) not in result.stderr
