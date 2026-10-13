"""Verify locked artifacts against primary PyPI metadata; run only in Docker."""

from __future__ import annotations

import json
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

CUTOFF = datetime.fromisoformat("2026-04-08T00:00:00+00:00")


def verify(package: dict) -> str:
    name, version = package["name"], package["version"]
    source = f"https://pypi.org/pypi/{name}/{version}/json"
    with urlopen(source, timeout=30) as response:
        metadata = json.load(response)
    uploads = {item["url"]: item for item in metadata["urls"]}
    artifacts = list(package.get("wheels", []))
    if "sdist" in package:
        artifacts.append(package["sdist"])
    dates = []
    for artifact in artifacts:
        published = uploads[artifact["url"]]
        uploaded = datetime.fromisoformat(
            published["upload_time_iso_8601"].replace("Z", "+00:00")
        )
        if uploaded >= CUTOFF:
            raise ValueError(f"Post-cutoff locked artifact: {name} {version}")
        if published["digests"]["sha256"] != artifact["hash"].removeprefix("sha256:"):
            raise ValueError(f"Artifact hash mismatch: {name} {version}")
        dates.append(uploaded)
    if not dates:
        raise ValueError(f"No verifiable artifacts: {name} {version}")
    return (
        f"- [{name} {version}]({source}): latest locked artifact uploaded "
        f"{max(dates).isoformat()}."
    )


def main() -> None:
    lock = tomllib.loads(Path("uv.lock").read_text())
    packages = [p for p in lock["package"] if "registry" in p["source"]]
    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(verify, packages))
    report = "\n".join(
        [
            "# Dependency release evidence",
            "",
            "All locked artifacts were verified against PyPI JSON metadata",
            "(upload timestamp and SHA-256) with a cutoff of 2026-04-08 00:00 UTC,",
            "including all of 2026-04-07. Run the audit inside Docker:",
            "",
            "```bash",
            "docker compose --profile checks run --rm --no-deps checks python scripts/verify_dependency_dates.py",
            "```",
            "",
            "- [uv 0.10.0](https://github.com/astral-sh/uv/releases/tag/0.10.0): 2026-02-05.",
            "- [Python 3.12.13](https://www.python.org/downloads/release/python-31213/): 2026-03-03.",
            "- [Redis 7.4.7](https://github.com/redis/redis/releases/tag/7.4.7): 2025-11-02.",
            "",
            *rows,
            "",
            "Docker base tags pin component versions. OS layers/tags are not a",
            "historical snapshot and may be rebuilt upstream; dependency artifact",
            "hashes are frozen in uv.lock. The PyPI audit requires network access.",
            "",
        ]
    )
    destination = Path("docs/specs/dependencies.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report)
    print(f"Verified {len(packages)} packages and every locked artifact before cutoff")


if __name__ == "__main__":
    main()
