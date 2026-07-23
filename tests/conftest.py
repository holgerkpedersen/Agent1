import pytest
from pathlib import Path

# Ensure pyfakefs plugin is explicitly loaded for reliable test isolation
pytest_plugins = ("pytest_fakefs",)


# ---------------------------------------------------------------------------
# 1️⃣ pyfakefs Configuration
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def fake_filesystem(fs):  # type: ignore[name-defined]
    """
    Automatically mount a clean, deterministic fake filesystem for every test.
    Prevents cross-test contamination and ensures predictable I/O behavior.
    """
    yield fs


# ---------------------------------------------------------------------------
# 2️⃣ pytest-asyncio Configuration
# ---------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    """
    Apply global pytest-asyncio settings programmatically.
    - Enforces 'function' scope for the default event loop fixture to guarantee 
      per-test async isolation and proper cleanup.
    - Sets asyncio mode to 'auto' so async test functions run without explicit markers.
    """
    config.addinivalue_line("asyncio_default_fixture_loop_scope", "function")
    config.addinivalue_line("addopts", "--asyncio-mode=auto")


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Force asyncio as the backend for anyio-compatible async fixtures/tests."""
    return "asyncio"


# ---------------------------------------------------------------------------
# 📦 Reusable Test Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_workspace_root(tmp_path: Path) -> Path:
    """Provide an isolated temporary directory simulating the agent's workspace sandbox."""
    ws = tmp_path / "agent_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws