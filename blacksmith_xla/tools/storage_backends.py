# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import List


class StorageBackend(ABC):
    """Abstract base class for storage backends"""

    @abstractmethod
    def save(self, local_path: str, remote_path: str = None):
        """Upload file to remote storage"""
        pass

    @abstractmethod
    def load(self, remote_path: str, local_path: str = None):
        """Download file from remote storage"""
        pass

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Check if file exists in remote storage"""
        pass

    @abstractmethod
    def delete(self, remote_path: str):
        """Delete file from remote storage"""
        pass

    @abstractmethod
    def list_files(self, remote_dir: str) -> List[str]:
        """List files in remote directory"""
        pass
