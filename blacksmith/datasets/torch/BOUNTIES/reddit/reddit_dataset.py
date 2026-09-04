# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from pickle import UnpicklingError
from zipfile import BadZipFile, ZipFile
from zlib import error as ZlibError

from filelock import FileLock
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader

from blacksmith.datasets.torch.torch_dataset import BaseDataset, TestDataLoaderWrapper
from blacksmith.tools.templates.configs import TrainingConfig

REDDIT_ARCHIVE_SIZE = 1_397_962_821
REDDIT_ARCHIVE_MEMBERS = ("reddit_graph.npz", "reddit_data.npz")


def _valid_reddit_archive(path: Path) -> bool:
    """Return whether a cached DGL Reddit download is complete and intact."""
    try:
        if path.stat().st_size != REDDIT_ARCHIVE_SIZE:
            return False
        with ZipFile(path) as archive:
            if tuple(archive.namelist()) != REDDIT_ARCHIVE_MEMBERS:
                return False
            return archive.testzip() is None
    except (BadZipFile, OSError, ZlibError):
        return False


def _recoverable_processed_error(error: Exception) -> bool:
    """Identify load failures caused by an interrupted processed-file write."""
    if isinstance(error, (EOFError, UnpicklingError)):
        return True
    message = str(error)
    return isinstance(error, RuntimeError) and (
        "PytorchStreamReader" in message or "failed finding central directory" in message
    )


class RedditDataset(BaseDataset):
    def __init__(self, config: TrainingConfig, split: str = "train") -> None:
        self._loaders: dict[str, NeighborLoader] = {}
        super().__init__(config=config, split=split)

    def _prepare_dataset(self) -> None:
        root = Path(self.config.dataset_root)
        root.parent.mkdir(parents=True, exist_ok=True)

        # PyG reuses a same-named ZIP without validating it. Serialize shared
        # cache initialization and remove an interrupted/corrupt download before
        # handing control to PyG, which downloads, extracts, and processes it.
        with FileLock(f"{root}.lock"):
            archive_path = root / "raw" / "reddit.zip"
            if archive_path.exists():
                archive_is_valid = _valid_reddit_archive(archive_path)
                # PyG deletes the ZIP only after extracting both members. A
                # leftover archive therefore means extraction may have been
                # interrupted; clear both raw targets and retry atomically.
                for member in REDDIT_ARCHIVE_MEMBERS:
                    (archive_path.parent / member).unlink(missing_ok=True)
                if not archive_is_valid:
                    archive_path.unlink(missing_ok=True)
            try:
                self.dataset = Reddit(root=str(root))
            except (EOFError, RuntimeError, UnpicklingError) as error:
                processed_path = root / "processed" / "data.pt"
                if not processed_path.exists() or not _recoverable_processed_error(error):
                    raise
                processed_path.unlink()
                self.dataset = Reddit(root=str(root))
            self.data = self.dataset[0]

    def _get_dataloader(self) -> NeighborLoader:
        return self._get_neighbour_loader(self.split)

    def _get_neighbour_loader(self, split: str) -> NeighborLoader:
        masks = {
            "train": self.data.train_mask,
            "val": self.data.val_mask,
            "test": self.data.test_mask,
        }
        if split not in masks:
            valid_splits = ", ".join(masks)
            raise ValueError(f"Unknown Reddit split '{split}'. Expected one of: {valid_splits}.")

        if split in self._loaders:
            return self._loaders[split]

        batch_size = self.config.batch_size if split == "train" else self.config.val_batch_size
        self._loaders[split] = NeighborLoader(
            self.data,
            num_neighbors=self.config.num_neighbors,
            batch_size=batch_size,
            input_nodes=masks[split],
            shuffle=(split == "train"),
            subgraph_type="directional",
        )
        return self._loaders[split]

    def get_neighbour_loader(self, split: str | None = None) -> NeighborLoader | TestDataLoaderWrapper:
        requested_split = self.split if split is None else split
        loader = self._get_neighbour_loader(requested_split)
        # Applies the shared CI test-mode step limit (config.test_config.max_steps_per_epoch)
        # to every split, since BaseDataset.get_dataloader() is bypassed here.
        return self._prepare_test_dataloader(loader)

    @property
    def num_features(self) -> int:
        return self.dataset.num_features

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    @property
    def num_nodes(self) -> int:
        return self.data.num_nodes

    @property
    def num_edges(self) -> int:
        return self.data.num_edges

    @property
    def train_nodes(self) -> int:
        return int(self.data.train_mask.sum())

    @property
    def val_nodes(self) -> int:
        return int(self.data.val_mask.sum())

    @property
    def test_nodes(self) -> int:
        return int(self.data.test_mask.sum())
