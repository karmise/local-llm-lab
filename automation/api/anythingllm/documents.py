"""Document storage operations of the AnythingLLM developer API."""

import json
from pathlib import Path

from requests import Response

from api.base_client import BaseClient
from reporting.steps import step


class DocumentsApi(BaseClient):
    """Document folders and uploads; indexing a document into a workspace belongs to `WorkspacesApi`."""
    def create_folder(self, name: str) -> Response:
        return self._request("POST", "document/create-folder", json={"name": name})

    def get_folder(self, name: str) -> Response:
        return self._request("GET", f"documents/folder/{self._segment(name)}")

    @step("API: delete test document folder")
    def delete_folder(self, name: str, *, timeout: float = 180) -> Response:
        return self._request("DELETE", "document/remove-folder", json={"name": name}, timeout=timeout)

    @step("API: upload document")
    def upload(
            self, path: Path, folder: str, *, filename: str | None = None, title: str | None = None,
            timeout: float = 180) -> Response:
        """Upload ``path`` into ``folder``. The document title, which the model sees as the source name, is the
        uploaded file name unless ``title`` is given."""
        metadata = {"metadata": json.dumps({"title": title})} if title else None
        with path.open("rb") as document:
            file = (filename or path.name, document, "text/plain")
            return self._request(
                    "POST", f"document/upload/{self._segment(folder)}", files={"file": file}, data=metadata,
                    timeout=timeout)
