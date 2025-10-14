import os
import shutil
from typing import List
from abc import ABC, abstractmethod

class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    def save(self, local_path: str, remote_path: str):
        """Upload file to a storage"""
        pass
    
    @abstractmethod
    def load(self, remote_path: str, local_path: str):
        """Download file from a storage"""
        pass
    
    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Check if file exists in a storage"""
        pass
    
    @abstractmethod
    def delete(self, remote_path: str):
        """Delete file from a storage"""
        pass
    
    @abstractmethod
    def list_files(self, remote_dir: str) -> List[str]:
        """List files in a directory"""
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage backend"""

    def save(self, local_path: str, remote_path: str):
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        shutil.copy2(local_path, remote_path)
    
    def load(self, remote_path: str, local_path: str):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(remote_path, local_path)
    
    def exists(self, remote_path: str) -> bool:
        return os.path.exists(remote_path)
    
    def delete(self, remote_path: str):
        if os.path.exists(remote_path):
            os.remove(remote_path)
    
    def list_files(self, remote_dir: str) -> List[str]:
        if not os.path.exists(remote_dir):
            return []
        return [f for f in os.listdir(remote_dir) if os.path.isfile(os.path.join(remote_dir, f))]