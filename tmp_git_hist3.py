"""Scratch probe: find the [UNVERIFIED] line in the git blob (deleted after)."""
import subprocess

r = subprocess.run(
    ["git", "log", "--follow", "-p", "-S", "_safe_path is defined at agent.py:554",
     "--", "CHANGES.md"],
    capture_output=True, text=True, encoding="utf-8", errors="replace",
)
out = r.stdout or ""
print("blocks:", len(out.split("commit ")) - 1)
for block in out.split("commit ")[1:]:
    for line in block.split("\n"):
        if "[UNVERIFIED]" in line and "line 86" in line:
            print(repr(line[:220]))
