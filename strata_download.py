"""Download the IST-DASLab / unsloth DeepSeek-V4.1-Flash GGUF sets to a local folder.

Scope: a plain GGUF fetch of an official set that Strata can launch.

Strata (https://github.com/Niko1221/Strata) added DeepSeek-V4.1-Flash as the
``deepseek41`` family in 0.2.4: ``setup.py`` carries ``DSV41_FILES`` /
``DSV41_ISTA_FILES`` and ``MULTI_FILE_FAMILIES["deepseek41"]`` with the models
``IST-DAS-1.75bpw``, ``IST-DAS-2.00bpw`` and ``IST-DAS-2.25bpw`` -- the three
sets laid out here -- and ``docs/DEEPSEEK41.md`` documents them.  See
``run_hint()`` for the two commands that turn a finished folder into a running
model.

A finished folder is *not* launchable as it stands: ``strata launch --model-path``
wants a packed model, not loose GGUF shards, so ``setup.sh --family deepseek41
--gguf-dir <folder>`` must pack the shards first.  Strata also needs its separate
``strata-dsv41`` engine next to the Strata engine; that binary is not part of this
fetch either.

The file names, sizes and SHA-256 digests below were read from the Hugging Face
Hub API at the pinned revisions and are re-checked against it by
``tests/test_strata_download.py::test_pinned_sizes_and_digests_match_the_hub``.

Layout written (flat, one folder per set)::

    D:\\models\\deepseek-v4.1-flash-ista-2.00bpw\\
        DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf
        ... 00002 / 00003 / 00004
        DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
        dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf

Usage::

    python strata_download.py --list
    python strata_download.py --set ista-2.00bpw --dest D:\\models --plan
    python strata_download.py --set ista-2.00bpw --dest D:\\models

The download is resumable: run the same command again after an interruption and
only the missing bytes are fetched.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

GB = 1_000_000_000

# Repositories and the exact revisions Strata pins (setup.py: HF_REVISIONS).
ISTA_REPO = "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF"
ISTA_REVISION = "cb02422c900df489cea5833e8a92171224059b3c"  # 2026-10-24, the final sets
UNSLOTH_REPO = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
UNSLOTH_REVISION = "f924b94f19c1fea02c06a65c8475b0a45e2b88b1"

# The one 4-bit Engram (n-gram) table every official set uses.
ENGRAM_FILE = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
# The DSpark speculative drafter, shared by every set.
DSPARK_FILE = "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"
DSPARK_PATH = DSPARK_FILE  # root of the Unsloth repository
# Only the GGUF metadata of Unsloth's first shard is read (pack_gguf --engram-meta).
META_FILE = "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf"
META_PATH = f"UD-Q8_K_XL/{META_FILE}"

SHARD_TEMPLATE = "DeepSeek-V4.1-Flash-1017-GSQ-RCO-{quant}-{index:05d}-of-00004.gguf"

# name -> (bytes, sha256); copied from Strata setup.py DSV41_ISTA_FILES /
# DSV41_FILES. The Hub's LFS digests match these values.
ISTA_FILES: dict[str, tuple[int, str]] = {
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf": (
        39358830464,
        "6e3f0d5ff53d00428ec32c079db9e8255cb410ccaa435e5342aaa8a47fe1339d",
    ),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00002-of-00004.gguf": (
        39176113248,
        "34d68c99fdae9948f9f0af0dafe308e805ce0a78d2c48641b24cc57242fdc0af",
    ),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00003-of-00004.gguf": (
        39997334720,
        "a08468c83955e0c4d3a9eb3f7b86c45b7bbac55d99c4640039a296a8a4b327ea",
    ),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00004-of-00004.gguf": (
        2188841824,
        "b05f26a92325aa6845cebcb3ec1f93d2e65ebf89bdc3f805695960d2f0681832",
    ),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf": (
        39535776992,
        "4c65307992ab586c527ca1f6f64062aa96a33b23c02b7fed103f45f75e854798",
    ),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00002-of-00004.gguf": (
        39494617536,
        "933461a974fcccee741bf2f9798201e72ab4aa3f8a952be56f849896ca6ea30e",
    ),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00003-of-00004.gguf": (
        39423838496,
        "bbf91fa346a85ebaf32dd4fd5c456b534d780d381e54f46ac90783a481373240",
    ),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00004-of-00004.gguf": (
        36771591232,
        "e5b2f8781fab688cd4bc3bc7a69651b5dbb90ed94965657c6bead8bb4774f3cf",
    ),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf": (
        39075714624,
        "d46efb67ccffd8fb5d7f18b423896381b9e06d8491e161c370ddf3b05b07e050",
    ),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00004.gguf": (
        39388449504,
        "cec1b2c3da429dc768b8bb9bc4dc4403d81f74d34db660a6a34ee3e555bf2c70",
    ),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00004.gguf": (
        39246891616,
        "9d9c2140fcb134a0b9a74dedd0e5167e68077af6d08ce7f856bce112f0a35dd3",
    ),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00004.gguf": (
        20527837312,
        "9b790a2297ed3cdede379fc984f4a52bb95f3b23724c797af11d1fa6a69db059",
    ),
    ENGRAM_FILE: (
        110595290816,
        "eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a",
    ),
}

UNSLOTH_FILES: dict[str, tuple[int, str]] = {
    META_FILE: (
        5780589,
        "cc15d470afcb030558bbc4d8e2ccd840182e2e95de1a95198138efe63a5fd99f",
    ),
    DSPARK_FILE: (
        7976914240,
        "6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a",
    ),
}


@dataclass(frozen=True)
class SetSpec:
    """One set name: its quant tier and its human-readable label."""

    name: str
    quant: str  # IQ1_M (1.75 bpw), IQ2_XXS (2.00 bpw), IQ2_XS (2.25 bpw)
    label: str


SETS: dict[str, SetSpec] = {
    "ista-1.75bpw": SetSpec(
        "ista-1.75bpw", "IQ1_M",
        "1.75 bpw - smallest, weakest on prose",
    ),
    "ista-2.00bpw": SetSpec(
        "ista-2.00bpw", "IQ2_XXS", "2.00 bpw - the balanced set"
    ),
    "ista-2.25bpw": SetSpec(
        "ista-2.25bpw", "IQ2_XS",
        "2.25 bpw - best quality of the three",
    ),
}


@dataclass(frozen=True)
class RemoteFile:
    """One file to fetch: where it lives and how big it must be."""

    repo: str
    revision: str
    path_in_repo: str
    dest_name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class DownloadPlan:
    """What a set needs: the big files plus the folder they land in."""

    set_name: str
    files: tuple[RemoteFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / GB

    def dest_names(self) -> list[str]:
        return [f.dest_name for f in self.files]


def _remote(repo: str, revision: str, path_in_repo: str, name: str) -> RemoteFile:
    size, sha = (ISTA_FILES if repo == ISTA_REPO else UNSLOTH_FILES)[name]
    return RemoteFile(repo, revision, path_in_repo, name, size, sha)


def plan(set_name: str, dest_root: str | Path = "D:/models") -> DownloadPlan:
    """Build the download plan for one set name.

    Raises ``KeyError`` with the valid names when the set is unknown.
    """
    if set_name not in SETS:
        raise KeyError(
            f"unknown set {set_name!r}; choose one of {', '.join(sorted(SETS))}"
        )
    spec = SETS[set_name]
    shards = [
        SHARD_TEMPLATE.format(quant=spec.quant, index=i) for i in range(1, 5)
    ]
    files = [_remote(ISTA_REPO, ISTA_REVISION, name, name) for name in shards]
    files.append(_remote(ISTA_REPO, ISTA_REVISION, ENGRAM_FILE, ENGRAM_FILE))
    files.append(_remote(UNSLOTH_REPO, UNSLOTH_REVISION, META_PATH, META_FILE))
    files.append(_remote(UNSLOTH_REPO, UNSLOTH_REVISION, DSPARK_PATH, DSPARK_FILE))
    # Every file lands flat in one folder per set.
    return DownloadPlan(set_name, tuple(files))


def destination_folder(set_name: str, dest_root: str | Path = "D:/models") -> Path:
    """The folder a set is written to (flat, one folder per set)."""
    if set_name not in SETS:
        raise KeyError(
            f"unknown set {set_name!r}; choose one of {', '.join(sorted(SETS))}"
        )
    return Path(dest_root) / f"deepseek-v4.1-flash-{set_name}"


def free_gb(path: str | Path) -> float:
    """Free space on the volume that holds ``path`` (its nearest existing parent)."""
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / GB


# Keep room for the OS, the Strata engine and the model pack.
DISK_HEADROOM_GB = 10


def check_disk(
    download_plan: DownloadPlan,
    dest_root: str | Path,
    headroom_gb: float = DISK_HEADROOM_GB,
) -> tuple[bool, str]:
    """Say whether ``dest_root`` has room for the plan. Returns (ok, message)."""
    need = download_plan.total_gb + headroom_gb
    have = free_gb(dest_root)
    if have >= need:
        return True, f"disk: {have:.0f} GB free, {need:.0f} GB needed"
    return False, (
        f"disk: only {have:.0f} GB free on {dest_root}, but {need:.0f} GB is needed "
        f"({download_plan.total_gb:.0f} GB of files + {headroom_gb:.0f} GB headroom)"
    )


def missing_bytes(download_plan: DownloadPlan, dest_root: str | Path) -> int:
    """Bytes still to fetch, given the files already present at full size."""
    folder = destination_folder(download_plan.set_name, dest_root)
    todo = 0
    for f in download_plan.files:
        target = folder / f.dest_name
        have = target.stat().st_size if target.exists() else 0
        if have < f.size:
            todo += f.size - have
    return todo


def _download_one(f: RemoteFile, folder: Path) -> Path:
    """Fetch one file, resuming a partial one. Returns the flat local path.

    ``hf_hub_download`` mirrors the repository's directory structure under
    ``local_dir``, so the metadata shard arrives as
    ``UD-Q8_K_XL/<name>.gguf``, while ``strata launch --gguf-dir`` (and
    ``missing_bytes``/``download``) want every file directly in the folder.

    ``local_dir`` stays the set folder on purpose: that is where
    ``hf_hub_download`` keeps its resumable ``.cache/huggingface/download/``
    state, and pointing it at a staging folder would abandon a partially
    fetched shard and restart it from zero.  The finished file is moved up into
    the flat folder instead - a rename on the same volume, so no copy.
    """
    from huggingface_hub import hf_hub_download

    got = Path(
        hf_hub_download(
            repo_id=f.repo,
            filename=f.path_in_repo,
            revision=f.revision,
            local_dir=str(folder),
        )
    )
    target = folder / f.dest_name
    if got != target:
        target.unlink(missing_ok=True)
        shutil.move(str(got), str(target))
        # The move empties the mirrored directory but leaves it in place, so
        # the set folder would carry a stray subfolder next to the flat files.
        # Remove it upwards, stopping at the set folder and at the first
        # directory that still holds something (the .cache state, another
        # shard).  A failed rmdir only costs a cosmetic empty folder.
        parent = got.parent
        while parent != folder and folder in parent.parents:
            if any(parent.iterdir()):
                break
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return target


def download(
    download_plan: DownloadPlan,
    dest_root: str | Path,
    *,
    force: bool = False,
    progress=print,
) -> Path:
    """Download every file of the plan into the set folder.

    Resumable: a part-downloaded file continues where it stopped. Files
    already at their pinned size are skipped unless ``force`` is set.
    """
    folder = destination_folder(download_plan.set_name, dest_root)
    folder.mkdir(parents=True, exist_ok=True)

    ok, message = check_disk(download_plan, dest_root)
    progress(message)
    if not ok:
        raise RuntimeError(message)

    for index, f in enumerate(download_plan.files, 1):
        target = folder / f.dest_name
        if not force and target.exists() and target.stat().st_size == f.size:
            progress(f"[{index}/{len(download_plan.files)}] have  {f.dest_name}")
            continue
        progress(
            f"[{index}/{len(download_plan.files)}] fetch {f.dest_name} "
            f"({f.size / GB:.2f} GB) from {f.repo}"
        )
        path = _download_one(f, folder)
        got = path.stat().st_size if path.exists() else 0
        if got != f.size:
            raise RuntimeError(
                f"{f.dest_name}: {got:,} bytes on disk, {f.size:,} expected"
            )

    progress(f"done: {folder} ({download_plan.total_gb:.1f} GB)")
    return folder


def strata_model_name(set_name: str) -> str:
    """The ``--model`` key ``setup.sh`` uses for one of these sets."""
    if set_name not in SETS:
        raise KeyError(
            f"unknown set {set_name!r}; choose one of {', '.join(sorted(SETS))}"
        )
    # docs/DEEPSEEK41.md: setup.sh --family deepseek41 --model IST-DAS-2.00bpw
    return f"IST-DAS-{set_name.removeprefix('ista-')}"


def strata_launch_model(set_name: str) -> str:
    """The same set as ``strata launch --model`` spells it: lowercase."""
    # docs/DEEPSEEK41.md: strata launch --model ist-das-2.00bpw
    return strata_model_name(set_name).lower()


def run_hint(set_name: str, dest_root: str | Path = "D:/models") -> list[str]:
    """The Strata commands that pack and start the downloaded set.

    Two steps, because loose GGUF shards are not launchable: ``--gguf-dir``
    packs them (the Engram table and the DSpark drafter are looked for beside
    them, which is where this module puts them), then ``strata launch`` starts
    the packed model.  Both need the separate ``strata-dsv41`` engine.
    """
    folder = destination_folder(set_name, dest_root)
    return [
        # setup.py gguf_dir_shards(): --gguf-dir needs --model to know which
        # shard set sits in the folder; the Engram and drafter are found beside
        # the shards, which is where this module puts them.
        f'setup.sh --family deepseek41 --model {strata_model_name(set_name)} '
        f'--gguf-dir "{folder}"',
        f"strata launch --model {strata_launch_model(set_name)}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="set_name", default="ista-2.00bpw",
                        help="which set to fetch (default: ista-2.00bpw)")
    parser.add_argument("--dest", default="D:/models",
                        help="folder that holds the set folders (default: D:/models)")
    parser.add_argument("--list", action="store_true", help="list the sets and stop")
    parser.add_argument("--plan", action="store_true",
                        help="print the plan; download nothing")
    parser.add_argument("--force", action="store_true",
                        help="re-fetch files that are already complete")
    args = parser.parse_args(argv)

    if args.list:
        print(f"{'set':<14} {'download':>9}  {'quality':<38} folder")
        for name in sorted(SETS):
            p = plan(name, args.dest)
            print(f"{name:<14} {p.total_gb:8.1f}G  {SETS[name].label:<38} "
                  f"{destination_folder(name, args.dest)}")
        return 0

    try:
        p = plan(args.set_name, args.dest)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    folder = destination_folder(args.set_name, args.dest)
    ok, message = check_disk(p, args.dest)
    print(f"set:    {args.set_name} ({SETS[args.set_name].label})")
    print(f"folder: {folder}")
    print(f"files:  {len(p.files)}  total {p.total_gb:.1f} GB")
    for f in p.files:
        print(f"  {f.size / GB:8.2f} GB  {f.dest_name}")
    print(message)
    print(f"todo:   {missing_bytes(p, args.dest) / GB:.1f} GB left to fetch")
    if not ok:
        return 1
    if args.plan:
        return 0

    try:
        download(p, args.dest, force=args.force)
    except RuntimeError as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 1
    print()
    print("Strata (https://github.com/Niko1221/Strata) runs this set as its")
    print("deepseek41 family, and needs its separate strata-dsv41 engine. Pack the")
    print("loose shards, then start the packed model:")
    for step in run_hint(args.set_name, args.dest):
        print("  " + step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
