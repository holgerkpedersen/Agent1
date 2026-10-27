"""Tests for strata_download: the DeepSeek-V4.1-Flash GGUF fetch plan.

The module fetches one of the three official IST-DAS sets that Strata launches
as the deepseek41 family (see the module docstring).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import strata_download as sd


def test_three_official_sets_are_offered():
    assert set(sd.SETS) == {"ista-1.75bpw", "ista-2.00bpw", "ista-2.25bpw"}


@pytest.mark.parametrize(
    ("set_name", "quant"),
    [
        ("ista-1.75bpw", "IQ1_M"),
        ("ista-2.00bpw", "IQ2_XXS"),
        ("ista-2.25bpw", "IQ2_XS"),
    ],
)
def test_plan_uses_the_quant_tier_of_the_set(set_name: str, quant: str):
    plan = sd.plan(set_name)
    names = plan.dest_names()
    assert len([n for n in names if f"-{quant}-" in n]) == 4


def test_plan_has_four_shards_plus_the_shared_files():
    plan = sd.plan("ista-2.00bpw")
    names = plan.dest_names()
    assert len(names) == 7
    assert sd.ENGRAM_FILE in names
    assert sd.DSPARK_FILE in names
    assert sd.META_FILE in names
    # the four shards are numbered 1..4 and each appears exactly once
    for i in range(1, 5):
        suffix = f"IQ2_XXS-{i:05d}-of-00004.gguf"
        matches = [n for n in names if n.endswith(suffix)]
        assert len(matches) == 1, f"{suffix}: {matches}"


def test_plan_total_matches_the_documented_disk_figure():
    """Strata docs: 257 GB for 2.00 bpw, 274 GB for 2.25, 239 GB for 1.75."""
    expected = {"ista-1.75bpw": 239, "ista-2.00bpw": 257, "ista-2.25bpw": 274}
    for name, doc_gb in expected.items():
        total = sd.plan(name).total_gb
        # the docs round to whole GB; allow a gigabyte of slack either way
        assert abs(total - doc_gb) <= 1.0, f"{name}: {total:.1f} GB vs {doc_gb} GB"


def test_plan_files_come_from_the_pinned_revisions():
    plan = sd.plan("ista-2.00bpw")
    for f in plan.files:
        if f.repo == sd.ISTA_REPO:
            assert f.revision == sd.ISTA_REVISION
        else:
            assert f.repo == sd.UNSLOTH_REPO
            assert f.revision == sd.UNSLOTH_REVISION


def test_metadata_shard_and_drafter_keep_their_repository_paths():
    plan = sd.plan("ista-2.00bpw")
    by_name = {f.dest_name: f for f in plan.files}
    assert by_name[sd.META_FILE].path_in_repo == f"UD-Q8_K_XL/{sd.META_FILE}"
    assert by_name[sd.DSPARK_FILE].path_in_repo == sd.DSPARK_FILE


def test_unknown_set_names_the_valid_choices():
    with pytest.raises(KeyError) as exc:
        sd.plan("ista-9.99bpw")
    message = str(exc.value)
    assert "ista-2.00bpw" in message
    assert "unknown set" in message


def test_destination_folder_is_flat_and_per_set(tmp_path):
    folder = sd.destination_folder("ista-2.00bpw", tmp_path)
    assert folder == tmp_path / "deepseek-v4.1-flash-ista-2.00bpw"


def test_every_file_lands_directly_in_the_set_folder(tmp_path):
    plan = sd.plan("ista-2.00bpw")
    folder = sd.destination_folder(plan.set_name, tmp_path)
    for name in plan.dest_names():
        assert "/" not in name and "\\" not in name
    assert {f.dest_name for f in plan.files} == set(plan.dest_names())
    assert folder.name.startswith("deepseek-v4.1-flash-")


@pytest.fixture
def tiny(monkeypatch):
    """The real plan path, with every pinned file shrunk to a few bytes.

    The real sizes are tens of gigabytes: writing them to disk in a test would
    take minutes, so the size table is swapped for a tiny one. ``plan()`` and
    ``missing_bytes()`` still run their real code.
    """
    small_ista = {name: (4, sha) for name, (_size, sha) in sd.ISTA_FILES.items()}
    small_unsloth = {name: (6, sha) for name, (_size, sha) in sd.UNSLOTH_FILES.items()}
    monkeypatch.setattr(sd, "ISTA_FILES", small_ista)
    monkeypatch.setattr(sd, "UNSLOTH_FILES", small_unsloth)
    monkeypatch.setattr(sd, "free_gb", lambda _p: 4000.0)
    return sd.plan("ista-2.00bpw")


def _write_at(path, size: int) -> None:
    """Create `path` holding exactly `size` bytes (no sparse multi-GB files)."""
    path.write_bytes(b"\0" * size)


def test_missing_bytes_counts_incomplete_files(tmp_path, tiny):
    plan = tiny
    folder = sd.destination_folder(plan.set_name, tmp_path)
    folder.mkdir(parents=True)
    assert sd.missing_bytes(plan, tmp_path) == plan.total_bytes

    # a partial file only needs its own missing bytes
    first = plan.files[0]
    (folder / first.dest_name).write_bytes(b"x")
    assert sd.missing_bytes(plan, tmp_path) == plan.total_bytes - 1

    # a file at exactly the pinned size is skipped
    _write_at(folder / first.dest_name, first.size)
    assert sd.missing_bytes(plan, tmp_path) == plan.total_bytes - first.size


def test_check_disk_refuses_when_space_is_short(tmp_path, monkeypatch):
    plan = sd.plan("ista-2.00bpw")
    monkeypatch.setattr(sd, "free_gb", lambda _p: 50.0)
    ok, message = sd.check_disk(plan, tmp_path)
    assert ok is False
    assert "50 GB free" in message
    assert "needed" in message


def test_check_disk_accepts_when_space_is_there(tmp_path, monkeypatch):
    plan = sd.plan("ista-2.00bpw")
    monkeypatch.setattr(sd, "free_gb", lambda _p: 4000.0)
    ok, message = sd.check_disk(plan, tmp_path)
    assert ok is True
    assert "4000 GB free" in message


def test_download_stops_before_fetching_when_disk_is_short(tmp_path, monkeypatch):
    plan = sd.plan("ista-2.00bpw")
    monkeypatch.setattr(sd, "free_gb", lambda _p: 5.0)

    def boom(*_a, **_kw):  # pragma: no cover - it must not be called
        raise AssertionError("no file may be fetched when the disk is too small")

    monkeypatch.setattr(sd, "_download_one", boom)
    with pytest.raises(RuntimeError, match="only 5 GB free"):
        sd.download(plan, tmp_path, progress=lambda _m: None)


def test_download_skips_files_already_at_their_size(tmp_path, tiny, monkeypatch):
    plan = tiny
    folder = sd.destination_folder(plan.set_name, tmp_path)
    folder.mkdir(parents=True)

    calls: list[str] = []

    def fake_download(f: sd.RemoteFile, dest: Path) -> Path:
        calls.append(f.dest_name)
        target = dest / f.dest_name
        _write_at(target, f.size)
        return target

    monkeypatch.setattr(sd, "_download_one", fake_download)

    # the first file is already complete: only the other six are fetched
    first = plan.files[0]
    _write_at(folder / first.dest_name, first.size)

    out = sd.download(plan, tmp_path, progress=lambda _m: None)
    assert out == folder
    assert first.dest_name not in calls
    assert len(calls) == len(plan.files) - 1


def test_download_fetches_everything_when_folder_is_empty(
    tmp_path, tiny, monkeypatch
):
    plan = tiny
    fetched: list[str] = []

    def fake_download(f: sd.RemoteFile, dest: Path) -> Path:
        fetched.append(f.path_in_repo)
        target = dest / f.dest_name
        _write_at(target, f.size)
        return target

    monkeypatch.setattr(sd, "_download_one", fake_download)
    sd.download(plan, tmp_path, progress=lambda _m: None)
    assert fetched == [f.path_in_repo for f in plan.files]


def test_download_rejects_a_short_file(tmp_path, tiny, monkeypatch):
    plan = tiny

    def short_download(f: sd.RemoteFile, dest: Path) -> Path:
        target = dest / f.dest_name
        target.write_bytes(b"")  # truncated: 0 of f.size bytes
        return target

    monkeypatch.setattr(sd, "_download_one", short_download)
    with pytest.raises(RuntimeError, match="expected"):
        sd.download(plan, tmp_path, progress=lambda _m: None)


def test_run_hint_packs_the_folder_then_launches_the_set():
    steps = sd.run_hint("ista-2.25bpw", "D:/models")
    folder = sd.destination_folder("ista-2.25bpw", "D:/models")
    # Loose GGUF shards are not launchable: docs/DEEPSEEK41.md packs them with
    # setup.sh --gguf-dir (the Engram table and the DSpark drafter are looked
    # for beside them), and only then does `strata launch` start the model.
    assert len(steps) == 2
    pack, launch = steps
    # --gguf-dir needs --model: setup.py's gguf_dir_shards() picks the shard set
    # by model name, so a bare --gguf-dir cannot tell 2.00 from 2.25.
    assert pack == (
        'setup.sh --family deepseek41 --model IST-DAS-2.25bpw '
        f'--gguf-dir "{folder}"'
    )
    assert launch == "strata launch --model ist-das-2.25bpw"
    # --model-path would point at raw shards, not at a packed model.
    for step in steps:
        assert "--model-path" not in step
        assert "-00001-of-00004.gguf" not in step


@pytest.mark.parametrize(
    ("set_name", "model"),
    [
        ("ista-1.75bpw", "IST-DAS-1.75bpw"),
        ("ista-2.00bpw", "IST-DAS-2.00bpw"),
        ("ista-2.25bpw", "IST-DAS-2.25bpw"),
    ],
)
def test_hint_uses_the_strata_model_name_of_the_set(set_name: str, model: str):
    # Strata 0.2.4 added DeepSeek-V4.1-Flash as MULTI_FILE_FAMILIES["deepseek41"]
    # with these three keys (docs/DEEPSEEK41.md); the hint must name the family
    # and the key, or `strata launch` cannot find the set.
    assert sd.strata_model_name(set_name) == model
    # setup.sh takes the key as written; `strata launch --model` lowercases it.
    assert sd.strata_launch_model(set_name) == model.lower()
    pack, launch = sd.run_hint(set_name, "D:/models")
    assert "--family deepseek41" in pack
    assert f"--model {model}" in pack
    assert launch == f"strata launch --model {model.lower()}"


def test_strata_model_name_rejects_an_unknown_set():
    with pytest.raises(KeyError, match="unknown set"):
        sd.strata_model_name("nope")


def test_cli_list_prints_every_set(capsys):
    assert sd.main(["--list"]) == 0
    out = capsys.readouterr().out
    for name in sd.SETS:
        assert name in out
    assert "deepseek-v4.1-flash-ista-2.00bpw" in out


def test_cli_plan_downloads_nothing(tmp_path, monkeypatch, capsys):
    def boom(*_a, **_kw):  # pragma: no cover - must not be reached
        raise AssertionError("--plan must not download")

    monkeypatch.setattr(sd, "download", boom)
    monkeypatch.setattr(sd, "free_gb", lambda _p: 4000.0)
    code = sd.main(["--set", "ista-2.00bpw", "--dest", str(tmp_path), "--plan"])
    assert code == 0
    out = capsys.readouterr().out
    assert "ista-2.00bpw" in out
    assert not (tmp_path / "deepseek-v4.1-flash-ista-2.00bpw").exists()


def test_cli_unknown_set_exits_nonzero(tmp_path, capsys):
    assert sd.main(["--set", "nope", "--dest", str(tmp_path)]) == 2
    assert "unknown set" in capsys.readouterr().err


def test_cli_refuses_when_disk_is_short(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sd, "free_gb", lambda _p: 20.0)
    code = sd.main(["--set", "ista-2.00bpw", "--dest", str(tmp_path)])
    assert code == 1
    assert "only 20 GB free" in capsys.readouterr().out


# --- the real _download_one path: repository subfolders must be flattened ---
#
# hf_hub_download mirrors the repo structure under local_dir, so the metadata
# shard really arrives as UD-Q8_K_XL/<name>.gguf. The fakes above replace
# _download_one, so they never exercised that. These two tests stub only
# hf_hub_download and leave the module's own file handling in the path.


def _fake_hf_hub_download(sizes: dict[str, int]):
    """Stand-in for hf_hub_download that writes <local_dir>/<filename>."""

    def fake(*, repo_id, filename, revision, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * sizes[path.name])
        return str(path)

    return fake


def test_download_one_flattens_the_repository_subfolder(tmp_path, monkeypatch):
    import huggingface_hub

    plan = sd.plan("ista-2.00bpw")
    meta = next(f for f in plan.files if f.dest_name == sd.META_FILE)
    # the shard genuinely lives in a subfolder upstream, so this is the case
    assert "/" in meta.path_in_repo

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", _fake_hf_hub_download({meta.dest_name: 4})
    )
    folder = tmp_path / "set"
    folder.mkdir()

    out = sd._download_one(meta, folder)

    assert out == folder / meta.dest_name
    assert out.is_file() and out.stat().st_size == 4
    # nothing is left behind under the mirrored path
    assert not list(folder.rglob(sd.META_FILE)) or [out] == list(
        folder.rglob(sd.META_FILE)
    )


def test_download_leaves_nothing_to_fetch_after_the_first_run(
    tmp_path, tiny, monkeypatch
):
    """Regression: a subfolder that survived the fetch left missing_bytes() > 0.

    Every rerun then re-fetched the same file, so the download could never
    report itself complete.
    """
    import huggingface_hub

    plan = tiny
    sizes = {f.dest_name: f.size for f in plan.files}
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", _fake_hf_hub_download(sizes)
    )

    folder = sd.download(plan, tmp_path, progress=lambda _m: None)

    assert {p.name for p in folder.iterdir() if p.is_file()} == set(sizes)
    assert sd.missing_bytes(plan, tmp_path) == 0


def test_flattening_leaves_no_empty_repository_folder(tmp_path, monkeypatch):
    """Regression: the move left the mirrored UD-Q8_K_XL/ folder behind, empty.

    A real fetch of ista-2.00bpw ended with a stray empty UD-Q8_K_XL/ beside the
    flat files, so the set folder did not look like the layout the module
    documents.
    """
    import huggingface_hub

    plan = sd.plan("ista-2.00bpw")
    meta = next(f for f in plan.files if f.dest_name == sd.META_FILE)
    assert "/" in meta.path_in_repo

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", _fake_hf_hub_download({meta.dest_name: 4})
    )
    folder = tmp_path / "set"
    folder.mkdir()

    sd._download_one(meta, folder)

    assert [p.name for p in folder.iterdir()] == [meta.dest_name]
    assert not [p for p in folder.rglob("*") if p.is_dir()]


def test_flattening_keeps_a_subfolder_that_still_holds_other_files(
    tmp_path, monkeypatch
):
    """Only the emptied folder goes; a sibling left in it must survive."""
    import huggingface_hub

    plan = sd.plan("ista-2.00bpw")
    meta = next(f for f in plan.files if f.dest_name == sd.META_FILE)

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", _fake_hf_hub_download({meta.dest_name: 4})
    )
    folder = tmp_path / "set"
    repo_dir = folder / meta.path_in_repo.rsplit("/", 1)[0]
    repo_dir.mkdir(parents=True)
    (repo_dir / "keep.txt").write_text("not ours to remove")

    sd._download_one(meta, folder)

    assert (folder / meta.dest_name).is_file()
    assert (repo_dir / "keep.txt").is_file()


# --- live checks against the Hub (network; opt out with SKIP_NETWORK_TESTS=1) ---


@pytest.mark.network
@pytest.mark.parametrize(
    ("repo", "revision", "table"),
    [
        (sd.ISTA_REPO, sd.ISTA_REVISION, "ISTA_FILES"),
        (sd.UNSLOTH_REPO, sd.UNSLOTH_REVISION, "UNSLOTH_FILES"),
    ],
)
def test_pinned_sizes_and_digests_match_the_hub(repo, revision, table):
    """Every pinned size and SHA-256 must equal the Hub's LFS metadata.

    A mismatch means a pinned revision changed or a pin is wrong, and the
    download would fetch bytes Strata would not accept.
    """
    import os

    if os.environ.get("SKIP_NETWORK_TESTS"):
        pytest.skip("SKIP_NETWORK_TESTS is set")

    import requests

    entries = requests.get(
        f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1",
        timeout=60,
    ).json()
    paths = {e["path"]: e for e in entries}

    want = getattr(sd, table)
    checked = 0
    for name, (size, sha256) in want.items():
        entry = paths.get(name) or paths.get(f"UD-Q8_K_XL/{name}")
        assert entry is not None, f"{name} is not in {repo}@{revision}"
        assert entry["size"] == size, f"{name}: size {entry['size']:,} != {size:,}"
        oid = (entry.get("lfs") or {}).get("oid") or entry.get("oid")
        assert oid == sha256, f"{name}: hub digest {oid} != pinned {sha256}"
        checked += 1
    assert checked == len(want)
