"""File search operations for agent."""
import asyncio
import os
import platform

from agent_core import to_windows_path


class FileSearcher:
    """Handles file searching with platform-appropriate commands.
    
    Extracted from Agent class to separate search concerns.
    """
    
    def __init__(self, workspace: str = None):
        self.workspace = workspace
    
    async def search(self, query: str, path: str) -> str:
        """Search for text pattern in files."""
        local_path = self._safe_path(path)
        
        results = await self._search_files(query, local_path)
        
        if not results:
            return "No matches found"
        
        return "\n".join(results)

    def _safe_path(self, path: str) -> str:
        """Normalize path for search."""
        if path.startswith(("./", ".\\")):
            path = path[2:]
        return to_windows_path(path)

    async def _search_files(self, query: str, local_path: str) -> list[str]:
        """Search files with platform-appropriate command and fallback."""
        results = []

        if platform.system() == "Windows":
            try:
                proc = await asyncio.create_subprocess_exec(
                    "findstr", "/S", "/N", "/C:" + query, local_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await proc.communicate()

                if proc.returncode == 0:
                    results = stdout.decode().splitlines()
                else:
                    results = await self._fallback_search(query, local_path)

            except FileNotFoundError:
                results = await self._fallback_search(query, local_path)
        else:
            proc = await asyncio.create_subprocess_exec(
                "grep", "-rn", query, local_path,
                stdout=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            results = stdout.decode().splitlines()

        return results

    async def _fallback_search(self, query: str, path: str) -> list[str]:
        """Fallback search using Python's os.walk with chunked reading."""
        results = []
        chunk_size = 8192

        for root, dirs, files in os.walk(path):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            if query in chunk:
                                results.append(filepath)
                                break
                except Exception:
                    pass

        return results
