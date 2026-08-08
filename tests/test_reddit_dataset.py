# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pytest

try:
    from blacksmith.datasets.torch.BOUNTIES.reddit import reddit_dataset
    from blacksmith.datasets.torch.BOUNTIES.reddit.reddit_dataset import (
        RedditDataset,
        _valid_reddit_archive,
    )
except ModuleNotFoundError as error:
    if error.name and error.name.startswith(("filelock", "torch_geometric")):
        pytest.skip("Reddit dataset tests require filelock and torch_geometric", allow_module_level=True)
    raise

pytestmark = [
    pytest.mark.push,
    pytest.mark.n300,
    pytest.mark.torch,
    pytest.mark.single_chip,
    pytest.mark.pyg,
]


def test_reddit_archive_validation_checks_size_and_members(tmp_path, monkeypatch) -> None:
    archive_path = tmp_path / "reddit.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("reddit_graph.npz", b"graph-data")
        archive.writestr("reddit_data.npz", b"node-data")

    monkeypatch.setattr(reddit_dataset, "REDDIT_ARCHIVE_SIZE", archive_path.stat().st_size)
    assert _valid_reddit_archive(archive_path)

    with ZipFile(archive_path, "a") as archive:
        archive.writestr("unexpected.txt", b"unexpected")
    monkeypatch.setattr(reddit_dataset, "REDDIT_ARCHIVE_SIZE", archive_path.stat().st_size)
    assert not _valid_reddit_archive(archive_path)


def test_reddit_archive_validation_checks_crc(tmp_path, monkeypatch) -> None:
    archive_path = tmp_path / "reddit.zip"
    with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("reddit_graph.npz", b"graph-data")
        archive.writestr("reddit_data.npz", b"node-data")
        graph_info = archive.getinfo("reddit_graph.npz")

    data_offset = graph_info.header_offset + 30 + len(graph_info.filename.encode()) + len(graph_info.extra)
    with archive_path.open("r+b") as archive_file:
        archive_file.seek(data_offset)
        original = archive_file.read(1)
        archive_file.seek(data_offset)
        archive_file.write(bytes([original[0] ^ 0xFF]))

    monkeypatch.setattr(reddit_dataset, "REDDIT_ARCHIVE_SIZE", archive_path.stat().st_size)
    assert not _valid_reddit_archive(archive_path)


def test_reddit_dataset_discards_interrupted_download(tmp_path, monkeypatch) -> None:
    root = tmp_path / "Reddit"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    interrupted_archive = raw_dir / "reddit.zip"
    interrupted_archive.write_bytes(b"partial download")

    graph = object()

    class FakeReddit:
        def __init__(self, root: str) -> None:
            assert root == str(tmp_path / "Reddit")
            assert not interrupted_archive.exists()

        def __getitem__(self, index: int) -> object:
            assert index == 0
            return graph

    monkeypatch.setattr(reddit_dataset, "Reddit", FakeReddit)
    dataset = RedditDataset(SimpleNamespace(dataset_root=str(root)))

    assert dataset.data is graph
    assert not interrupted_archive.exists()


def test_reddit_dataset_reextracts_after_interrupted_extraction(tmp_path, monkeypatch) -> None:
    root = tmp_path / "Reddit"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    archive_path = raw_dir / "reddit.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("reddit_graph.npz", b"complete graph")
        archive.writestr("reddit_data.npz", b"complete data")

    for member in reddit_dataset.REDDIT_ARCHIVE_MEMBERS:
        (raw_dir / member).write_bytes(b"partial extraction")
    monkeypatch.setattr(reddit_dataset, "REDDIT_ARCHIVE_SIZE", archive_path.stat().st_size)

    graph = object()

    class FakeReddit:
        def __init__(self, root: str) -> None:
            assert archive_path.exists()
            for member in reddit_dataset.REDDIT_ARCHIVE_MEMBERS:
                assert not (raw_dir / member).exists()

        def __getitem__(self, index: int) -> object:
            assert index == 0
            return graph

    monkeypatch.setattr(reddit_dataset, "Reddit", FakeReddit)
    dataset = RedditDataset(SimpleNamespace(dataset_root=str(root)))

    assert dataset.data is graph


def test_reddit_dataset_reprocesses_interrupted_processed_file(tmp_path, monkeypatch) -> None:
    root = tmp_path / "Reddit"
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True)
    interrupted_processed = processed_dir / "data.pt"
    interrupted_processed.write_bytes(b"partial processed graph")

    graph = object()
    attempts = 0

    class FakeReddit:
        def __init__(self, root: str) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("PytorchStreamReader failed finding central directory")
            assert not interrupted_processed.exists()

        def __getitem__(self, index: int) -> object:
            assert index == 0
            return graph

    monkeypatch.setattr(reddit_dataset, "Reddit", FakeReddit)
    dataset = RedditDataset(SimpleNamespace(dataset_root=str(root)))

    assert attempts == 2
    assert dataset.data is graph
