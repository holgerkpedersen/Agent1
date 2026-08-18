"""File system operations for agent."""
import os

from agent_core.path_utils import resolve_path, safe_path


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
        local_path = self.safe_path(path)

        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error reading file: {e}"

    async def write(self, path: str, content: str) -> str:
        """Write content to file."""
        local_path = self.safe_path(path)

        try:
            dir_name = os.path.dirname(local_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return f"Successfully wrote to {path}"
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    async def apply_patch(self, path: str, find: str, replace: str) -> str:
        """Apply find-and-replace patch to file."""
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
            return f"Permission denied: {path}"
        except Exception as e:
            return f"Error applying patch: {e}"

    async def edit(self, path: str, content: str) -> str:
        """Overwrite file with new content."""
        local_path = self.safe_path(path)

        try:
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return f"Successfully edited {path}"
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            return f"Error editing file: {e}"

    async def list_files(self, path: str, pattern: str = "*") -> str:
        """List directory entries (dirs marked with /), one per line."""
        local_path = self.safe_path(path)
        try:
            entries = os.listdir(local_path)
        except OSError as e:
            return f"List error: {e}"
        lines = []
        for entry in sorted(entries)[:50]:
            full = os.path.join(local_path, entry)
            suffix = "/" if os.path.isdir(full) else ""
            lines.append(f"  {entry}{suffix}")
        return "\n".join(lines)

    async def delete(self, path: str) -> str:
        """Delete a file (or empty directory)."""
        local_path = self.safe_path(path)
        try:
            if os.path.isdir(local_path):
                os.rmdir(local_path)
            else:
                os.remove(local_path)
            return f"Deleted {path}"
        except OSError as e:
            return f"Delete error: {e}"
