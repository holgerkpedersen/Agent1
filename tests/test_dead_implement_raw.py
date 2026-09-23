from pathlib import Path
import pytest

def test_implement_raw_module_does_not_exist():
    """Verify that the dead module _implement_raw.py has been removed."""
    assert not Path("agent_core/commands/_implement_raw.py").exists()

def test_no_references_to_dead_symbols():
    """
    Scan core directories for any remaining references to dead symbols 
    to ensure complete removal and prevent regressions.
    """
    search_tokens = ["_implement_raw", "RawImplementCommand"]
    # Exclude the test file itself from scanning to avoid self-reference false positives
    test_file_path = Path(__file__).resolve()

    # Explicitly target core directories as per plan to avoid false positives in gitignored/temp dirs
    target_dirs = [
        "agent_core",
        "agent.py",
        "tools",
        "scripts",
        "harnessfix",
        "benchmarks",
        "tests"
    ]
    
    found_references = []

    for target in target_dirs:
        path = Path(target)
        if not path.exists():
            continue
            
        # If it's a file, check it directly; if directory, walk it
        files_to_check = [path] if path.is_file() else path.rglob("*.py")
        
        for file_path in files_to_check:
            if file_path.resolve() == test_file_path:
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                for token in search_tokens:
                    if token in content:
                        found_references.append(f"{file_path}:{token}")
            except Exception as e:
                # Skip files that can't be read (e.g. permission issues)
                continue

    assert not found_references, f"Found dead symbols in: {found_references}"
