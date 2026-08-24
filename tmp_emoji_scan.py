"""Scratch probe: run the scan but catch everything (deleted after)."""
import subprocess
import traceback
import unicodedata

exts = (".py", ".md", ".json", ".txt", ".toml", ".cfg", ".ini", ".yml", ".yaml")
skip = ("chat_history.json", "agent_memory.json")


def is_symbolic(ch):
    if ord(ch) < 0x80:
        return False
    cat = unicodedata.category(ch)
    if cat in ("So", "Sk"):
        return True
    return 0x1F000 <= ord(ch) <= 0x1FAFF or 0x2600 <= ord(ch) <= 0x27BF


try:
    r = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    files = [p for p in (r.stdout or "").split("\n") if p.strip()]
    print("files:", len(files), flush=True)

    hits = {}
    for path in files:
        if not path.endswith(exts) or any(s in path for s in skip):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        for ln, line in enumerate(data.split("\n"), 1):
            found = sorted({c for c in line if is_symbolic(c)})
            if found:
                hits.setdefault(path, []).append((ln, "".join(found)))

    print("scanned; files with hits:", len(hits), flush=True)
except BaseException:
    traceback.print_exc()
    sys_exit_code = 1
else:
    sys_exit_code = 0

with open("_tmp_scan_out.txt", "w", encoding="utf-8") as f:
    for path, rows in hits.items():
        f.write(f"{path}: {len(rows)} line(s)\n")
        for ln, chars in rows[:8]:
            f.write(f"   L{ln}: {chars!r}\n")
print("exit code:", sys_exit_code, flush=True)
