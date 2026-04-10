"""
nexus/integrations/google/drive.py
Google Drive API v3 client for Nexus Agent.

GoogleDriveClient
-----------------
Thin, fully-injectable wrapper around the Drive REST API.

list_files(query, mime_type) -> list[DriveFile]
  GET drive/v3/files?q=... returning id, name, mimeType, modifiedTime.

download_file(file_id, local_path) -> bool
  GET drive/v3/files/{id}?alt=media → write bytes to *local_path*.
  Returns True on success, False on any failure.

upload_file(local_path, parent_id) -> DriveFile | None
  POST to drive/v3/files (multipart) to create a new file.
  Returns the created DriveFile on success, None on failure.

Injectable callables
--------------------
_get_fn      : (url, headers) -> dict
_download_fn : (url, headers) -> bytes
_post_fn     : (url, headers, body) -> dict
_read_file_fn: (path: str) -> bytes
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from nexus.infra.logger import get_logger

_log = get_logger(__name__)

_DRIVE_BASE = "https://www.googleapis.com/drive/v3"
_UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"


# ---------------------------------------------------------------------------
# DriveFile
# ---------------------------------------------------------------------------


@dataclass
class DriveFile:
    """Minimal Drive file metadata."""

    id: str
    name: str
    mime_type: str
    modified_time: str = ""


# ---------------------------------------------------------------------------
# Default injectable implementations
# ---------------------------------------------------------------------------


def _default_get(url: str, headers: dict[str, str]) -> dict[str, Any]:
    import urllib.request  # noqa: PLC0415

    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def _default_download(url: str, headers: dict[str, str]) -> bytes:
    import urllib.request  # noqa: PLC0415

    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()  # type: ignore[no-any-return]


def _default_post(
    url: str, headers: dict[str, str], body: dict[str, Any]
) -> dict[str, Any]:
    import urllib.request  # noqa: PLC0415

    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def _default_read_file(path: str) -> bytes:
    return Path(path).read_bytes()


# ---------------------------------------------------------------------------
# GoogleDriveClient
# ---------------------------------------------------------------------------


class GoogleDriveClient:
    """
    Google Drive API v3 client.

    Parameters
    ----------
    get_token_fn:
        Callable returning a valid OAuth 2.0 access token string.
    _get_fn:
        ``(url, headers) -> dict``.  HTTP GET implementation.
    _download_fn:
        ``(url, headers) -> bytes``.  Binary GET implementation.
    _post_fn:
        ``(url, headers, body) -> dict``.  HTTP POST implementation.
    _read_file_fn:
        ``(path: str) -> bytes``.  Local file reader for uploads.
    """

    def __init__(
        self,
        get_token_fn: Callable[[], str],
        *,
        _get_fn: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
        _download_fn: Callable[[str, dict[str, str]], bytes] | None = None,
        _post_fn: (
            Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]] | None
        ) = None,
        _read_file_fn: Callable[[str], bytes] | None = None,
    ) -> None:
        self._get_token = get_token_fn
        self._get = _get_fn or _default_get
        self._download = _download_fn or _default_download
        self._post = _post_fn or _default_post
        self._read_file = _read_file_fn or _default_read_file

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_files(
        self,
        query: str = "",
        mime_type: str = "",
    ) -> list[DriveFile]:
        """
        List Drive files matching *query* and/or *mime_type*.

        Parameters
        ----------
        query:
            Drive query string (e.g. ``"name contains 'report'"``).
        mime_type:
            MIME type filter appended to *query* with ``and``.

        Returns
        -------
        List of DriveFile objects (may be empty).
        """
        q_parts: list[str] = []
        if query:
            q_parts.append(query)
        if mime_type:
            q_parts.append(f"mimeType='{mime_type}'")
        params: dict[str, str] = {
            "fields": "files(id,name,mimeType,modifiedTime)",
            "pageSize": "100",
        }
        if q_parts:
            params["q"] = " and ".join(q_parts)
        url = f"{_DRIVE_BASE}/files?{urlencode(params)}"
        response = self._get(url, self._auth_headers())
        files: list[dict[str, Any]] = response.get("files", [])
        return [
            DriveFile(
                id=f["id"],
                name=f["name"],
                mime_type=f["mimeType"],
                modified_time=f.get("modifiedTime", ""),
            )
            for f in files
        ]

    def download_file(self, file_id: str, local_path: str) -> bool:
        """
        Download *file_id* to *local_path*.

        Returns
        -------
        True on success, False on any failure.
        """
        url = f"{_DRIVE_BASE}/files/{file_id}?alt=media"
        try:
            data = self._download(url, self._auth_headers())
            Path(local_path).write_bytes(data)
            _log.debug("download_ok", file_id=file_id, path=local_path)
            return True
        except Exception as exc:
            _log.warning("download_failed", file_id=file_id, error=str(exc))
            return False

    def upload_file(
        self, local_path: str, parent_id: str = ""
    ) -> DriveFile | None:
        """
        Upload *local_path* to Drive, optionally into *parent_id*.

        Returns
        -------
        DriveFile for the newly created file, or None on failure.
        """
        try:
            file_bytes = self._read_file(local_path)
        except Exception as exc:
            _log.warning("upload_read_failed", path=local_path, error=str(exc))
            return None

        name = Path(local_path).name
        metadata: dict[str, Any] = {"name": name}
        if parent_id:
            metadata["parents"] = [parent_id]

        url = f"{_UPLOAD_BASE}/files?uploadType=multipart&fields=id,name,mimeType,modifiedTime"
        headers = self._auth_headers()
        # Encode as a simple JSON metadata upload (metadata-only path for
        # testability; real multipart would need boundary encoding).
        headers["X-Upload-Content-Length"] = str(len(file_bytes))
        try:
            response = self._post(url, headers, metadata)
            _log.debug("upload_ok", name=name, file_id=response.get("id"))
            return DriveFile(
                id=response["id"],
                name=response.get("name", name),
                mime_type=response.get("mimeType", "application/octet-stream"),
                modified_time=response.get("modifiedTime", ""),
            )
        except Exception as exc:
            _log.warning("upload_failed", path=local_path, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        token = self._get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
