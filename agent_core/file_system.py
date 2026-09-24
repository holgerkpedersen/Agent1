"""File system operations for agent."""
import logging
import os

from agent_core.exceptions import FileOperationError
from agent_core.path_utils import resolve_path, safe_path

logger = logging.getLogger(__name__)


class FileSystem:
    """Handles file I/O operations with path normalization.
    
    Extracted from Agent class to separate file system concerns.
    """
    
    def __init__(self, workspace: str):
        self.workspace = workspace
    
    def normalize_path(self, path: str) -> str:
        """Normalize and validate paths with security checks."""
        return resolve_path(path)

    def safe_path(self, path: str) -> str:
        """Validate and normalize path in one step."""
        return safe_path(path)

    async def read(self, path: str) -> str:
        """Read file contents."""
        if not isinstance(path, str) or not path.strip():
            raise TypeError("path must be a non-empty string")
        local_path = self.safe_path(path)

        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("File not found: %s", path)
            raise FileOperationError(f"File not found: {path}", path=path)
        except PermissionError:
            logger.warning("Permission denied reading: %s", path)
            raise FileOperationError(f"Permission denied: {path}", path=path)
        except OSError as e:
            logger.exception("OS error reading file: %s", path)
            raise FileOperationError(f"Error reading file: {e}", path=path) from e
        except Exception as e:
            logger.exception("Unexpected error reading file: %s", path)
            raise FileOperationError(f"Error reading file: {e}", path=path) from e

    async def write(self, path: str, content: str) -> str:
        """Write content to file."""
        if not isinstance(path, str) or not path.strip():
            raise TypeError("path must be a non-empty string")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        local_path = self.safe_path(path)

        try:
            dir_name = os.path.dirname(local_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return f"Successfully wrote to {path}"
        except PermissionError:
            logger.warning("Permission denied writing: %s", path)
            raise FileOperationError(f"Permission denied: {path}", path=path)
        except OSError as e:
            logger.exception("OS error writing file: %s", path)
            raise FileOperationError(f"Error writing file: {e}", path=path) from e
        except Exception as e:
            logger.exception("Unexpected error writing file: %s", path)
            raise FileOperationError(f"Error writing file: {e}", path=path) from e

    async def apply_patch(self, path: str, find: str, replace: str) -> str:
        """Apply find-and-replace patch to file."""
        if not isinstance(path, str) or not path.strip():
            raise TypeError("path must be a non-empty string")
        if not isinstance(find, str) or not find.strip():
            raise TypeError("find must be a non-empty string")
        if not isinstance(replace, str):
            raise TypeError("replace must be a string")
        local_path = self.safe_path(path)

        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if find not in content:
                return "Pattern not found in file"

            count = content.count(find)
            if count > 1:
                return f"Error: find text matches {count} locations. Add more context to make it unique."

            new_content = content.replace(find, replace, 1)

            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return "Patch applied successfully"
        except PermissionError:
            logger.warning("Permission denied patching: %s", path)
            raise FileOperationError(f"Permission denied: {path}", path=path)
        except FileNotFoundError:
            logger.warning("File not found for patching: %s", path)
            raise FileOperationError(f"File not found: {path}", path=path)
        except OSError as e:
            logger.exception("OS error applying patch: %s", path)
            raise FileOperationError(f"Error applying patch: {e}", path=path) from e
        except Exception as e:
            logger.exception("Unexpected error applying patch: %s", path)
            raise FileOperationError(f"Error applying patch: {e}", path=path) from e

    async def edit(self, path: str, content: str) -> str:
        """Overwrite file with new content."""
        if not isinstance(path, str) or not path.strip():
            raise TypeError("path must be a non-empty string")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        local_path = self.safe_path(path)

        try:
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return f"Successfully edited {path}"
        except PermissionError:
            logger.warning("Permission denied editing: %s", path)
            raise FileOperationError(f"Permission denied: {path}", path=path)
        except OSError as e:
            logger.exception("OS error editing file: %s", path)
            raise FileOperationError(f"Error editing file: {e}", path=path) from e
        except Exception as e:
            logger.exception("Unexpected error editing file: %s", path)
            raise FileOperationError(f"Error editing file: {e}", path=path) from e

    async def list_files(self, path: str, pattern: str = "*") -> str:
        """List directory entries (dirs marked with /), one per line."""
        if not isinstance(path, str) or not path.strip():
            raise TypeError("path must be a non-empty string")
        local_path = self.safe_path(path)
        try:
            entries = os.listdir(local_path)
        except FileNotFoundError:
            logger.warning("Directory not found: %s", path)
            raise FileOperationError(f"File not found: {path}", path=path)
        except PermissionError:
            logger.warning("Permission denied listing: %s", path)
            raise FileOperationError(f"Permission denied: {path}", path=path)
        except OSError as e:
            logger.exception("List error: %s", path)
            raise FileOperationError(f"List error: {e}", path=path) from e

        lines = []
        for entry in sorted(entries)[:50]:
            full = os.path.join(local_path, entry)
            suffix = "/" if os.path.isdir(full) else ""
            lines.append(f"  {entry}{suffix}")
        return "\n".join(lines)

    async def delete(self, path: str) -> str:
        """Delete a file (or empty directory)."""
        if not isinstance(path, str) or not path.strip():
            raise TypeError("path must be a non-empty string")
        local_path = self.safe_path(path)
        try:
            if os.path.isdir(local_path):
                os.rmdir(local_path)
            else:
                os.remove(local_path)
            return f"Deleted {path}"
        except FileNotFoundError:
            logger.warning("File not found for deletion: %s", path)
            raise FileOperationError(f"File not found: {path}", path=path)
        except PermissionError:
            logger.warning("Permission denied deleting: %s", path)
            raise FileOperationError(f"Permission denied: {path}", path=path)
        except OSError as e:
            logger.exception("Delete error: %s", path)
            raise FileOperationError(f"Delete error: {e}", path=path) from e


__all__: list[str] = [
    "FileSystem",
]
