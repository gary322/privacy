from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_4b3e1ae5d567ea44

from pathlib import Path


def test_repo_layout_and_imports_basic() -> None:
    repo_root = Path(__file__).resolve().parents[4]

    # Core directories from the build plan exist.
    must_exist = [
        repo_root / "research" / "uvcc" / "uvcc-spec",
        repo_root / "research" / "uvcc" / "uvcc-relay",
        repo_root / "research" / "uvcc" / "uvcc-party",
        repo_root / "research" / "uvcc" / "uvcc-verifier",
        repo_root / "research" / "uvcc" / "uvcc-contracts",
        repo_root / "research" / "uvcc" / "uvcc-client",
        repo_root / "research" / "uvcc" / "uvcc-demo",
    ]
    for p in must_exist:
        assert p.exists(), f"missing path: {p}"

    # Key entrypoints exist.
    assert (repo_root / "research" / "uvcc" / "uvcc-relay" / "relay_server.py").exists()
    assert (repo_root / "research" / "uvcc" / "uvcc-demo" / "run_demo.py").exists()

    # Python packages are importable with local sys.path wiring.
    import sys

    sys.path.insert(0, str(repo_root / "research" / "uvcc" / "uvcc-party"))
    sys.path.insert(0, str(repo_root / "research" / "uvcc" / "uvcc-verifier"))
    sys.path.insert(0, str(repo_root / "research" / "uvcc" / "uvcc-client"))

    import uvcc_party  # noqa: F401
    import uvcc_verifier  # noqa: F401
    import uvcc_client  # noqa: F401


