import shutil
from pathlib import Path

src = Path(".docs/2026-08-12_12-00-00/plan_proposed.md")
dst_dir = Path(".docs/2026-10-22_16-33-12")
dst_dir.mkdir(parents=True, exist_ok=True)
shutil.copyfile(src, dst_dir / "plan_proposed.md")
print("copied:", (dst_dir / "plan_proposed.md").stat().st_size, "bytes")
