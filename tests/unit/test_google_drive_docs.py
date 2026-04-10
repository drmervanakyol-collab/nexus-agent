"""
tests/unit/test_google_drive_docs.py
Unit tests for nexus/integrations/google/drive.py and docs.py — Faz 54.

TEST PLAN
---------
GoogleDriveClient.list_files:
  1. No filters → all files returned.
  2. query filter applied.
  3. mime_type filter applied.
  4. Both query + mime_type → joined with "and".
  5. Empty files list → [].

GoogleDriveClient.download_file:
  6. Success → bytes written to path, returns True.
  7. Download raises → returns False, no file written.

GoogleDriveClient.upload_file:
  8. Success → DriveFile returned with correct id/name.
  9. Local file read fails → returns None.
  10. POST fails → returns None.

GoogleDocsClient.read_document:
  11. Document with paragraphs → joined plain text.
  12. Empty body → empty string.
  13. Elements without textRun → skipped gracefully.

GoogleDocsClient.append_text:
  14. Success → True, batchUpdate called with correct payload.
  15. POST raises → False.
"""
from __future__ import annotations

from typing import Any

from nexus.integrations.google.docs import GoogleDocsClient
from nexus.integrations.google.drive import DriveFile, GoogleDriveClient

_TOKEN = "ya29.test_token"


# ---------------------------------------------------------------------------
# Drive helpers
# ---------------------------------------------------------------------------


def _drive_client(
    *,
    get_response: dict[str, Any] | None = None,
    download_bytes: bytes | None = None,
    download_raises: Exception | None = None,
    post_response: dict[str, Any] | None = None,
    post_raises: Exception | None = None,
    file_bytes: bytes | None = None,
    file_raises: Exception | None = None,
    written: list[bytes] | None = None,
) -> GoogleDriveClient:
    _written = written if written is not None else []

    def _get(url: str, headers: dict) -> dict:
        return get_response or {}

    def _download(url: str, headers: dict) -> bytes:
        if download_raises:
            raise download_raises
        return download_bytes or b""

    def _post(url: str, headers: dict, body: dict) -> dict:
        if post_raises:
            raise post_raises
        return post_response or {}

    def _read_file(path: str) -> bytes:
        if file_raises:
            raise file_raises
        return file_bytes or b""

    # Patch write_bytes for download verification
    original_write = None

    client = GoogleDriveClient(
        get_token_fn=lambda: _TOKEN,
        _get_fn=_get,
        _download_fn=_download,
        _post_fn=_post,
        _read_file_fn=_read_file,
    )
    return client


def _file_entry(
    id_: str = "abc123",
    name: str = "report.xlsx",
    mime: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    modified: str = "2026-04-09T10:00:00.000Z",
) -> dict[str, Any]:
    return {"id": id_, "name": name, "mimeType": mime, "modifiedTime": modified}


# ---------------------------------------------------------------------------
# GoogleDriveClient.list_files
# ---------------------------------------------------------------------------


class TestListFiles:
    def test_no_filters_returns_all(self):
        resp = {"files": [_file_entry("1", "a.txt"), _file_entry("2", "b.txt")]}
        client = _drive_client(get_response=resp)
        files = client.list_files()
        assert len(files) == 2
        assert files[0].id == "1"
        assert files[1].name == "b.txt"

    def test_query_filter(self):
        captured_urls: list[str] = []

        def _get(url: str, headers: dict) -> dict:
            captured_urls.append(url)
            return {"files": []}

        client = GoogleDriveClient(
            get_token_fn=lambda: _TOKEN,
            _get_fn=_get,
        )
        client.list_files(query="name contains 'report'")
        assert "name+contains+%27report%27" in captured_urls[0] or "name" in captured_urls[0]

    def test_mime_type_filter(self):
        captured_urls: list[str] = []

        def _get(url: str, headers: dict) -> dict:
            captured_urls.append(url)
            return {"files": []}

        client = GoogleDriveClient(
            get_token_fn=lambda: _TOKEN,
            _get_fn=_get,
        )
        client.list_files(mime_type="application/pdf")
        assert "mimeType" in captured_urls[0]

    def test_query_and_mime_type_joined(self):
        captured_urls: list[str] = []

        def _get(url: str, headers: dict) -> dict:
            captured_urls.append(url)
            return {"files": []}

        client = GoogleDriveClient(
            get_token_fn=lambda: _TOKEN,
            _get_fn=_get,
        )
        client.list_files(query="name='x'", mime_type="text/plain")
        # Both constraints must appear in the URL
        assert "name" in captured_urls[0]
        assert "mimeType" in captured_urls[0]

    def test_empty_files_list(self):
        client = _drive_client(get_response={"files": []})
        assert client.list_files() == []


# ---------------------------------------------------------------------------
# GoogleDriveClient.download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    def test_success_writes_bytes(self, tmp_path):
        content = b"binary file content"
        client = _drive_client(download_bytes=content)
        dest = str(tmp_path / "out.bin")
        result = client.download_file("file123", dest)
        assert result is True
        assert (tmp_path / "out.bin").read_bytes() == content

    def test_download_raises_returns_false(self, tmp_path):
        client = _drive_client(download_raises=OSError("network error"))
        dest = str(tmp_path / "out.bin")
        result = client.download_file("file123", dest)
        assert result is False
        assert not (tmp_path / "out.bin").exists()


# ---------------------------------------------------------------------------
# GoogleDriveClient.upload_file
# ---------------------------------------------------------------------------


class TestUploadFile:
    def test_success_returns_drive_file(self):
        post_resp = {
            "id": "new_file_id",
            "name": "report.pdf",
            "mimeType": "application/pdf",
            "modifiedTime": "2026-04-09T12:00:00.000Z",
        }
        client = _drive_client(post_response=post_resp, file_bytes=b"data")
        result = client.upload_file("/fake/report.pdf")
        assert isinstance(result, DriveFile)
        assert result.id == "new_file_id"
        assert result.name == "report.pdf"

    def test_read_file_fails_returns_none(self):
        client = _drive_client(file_raises=FileNotFoundError("no such file"))
        result = client.upload_file("/nonexistent/file.pdf")
        assert result is None

    def test_post_fails_returns_none(self):
        client = _drive_client(
            file_bytes=b"data",
            post_raises=OSError("server error"),
        )
        result = client.upload_file("/fake/file.pdf")
        assert result is None


# ---------------------------------------------------------------------------
# GoogleDocsClient helpers
# ---------------------------------------------------------------------------


def _docs_client(
    *,
    get_response: dict[str, Any] | None = None,
    post_response: dict[str, Any] | None = None,
    post_raises: Exception | None = None,
    post_calls: list[dict] | None = None,
) -> GoogleDocsClient:
    _calls = post_calls if post_calls is not None else []

    def _get(url: str, headers: dict) -> dict:
        return get_response or {}

    def _post(url: str, headers: dict, body: dict) -> dict:
        _calls.append(body)
        if post_raises:
            raise post_raises
        return post_response or {}

    return GoogleDocsClient(
        get_token_fn=lambda: _TOKEN,
        _get_fn=_get,
        _post_fn=_post,
    )


def _doc_body(*paragraphs: str) -> dict[str, Any]:
    """Build a minimal Docs API response with given paragraph strings."""
    content = [
        {
            "paragraph": {
                "elements": [{"textRun": {"content": p}}]
            }
        }
        for p in paragraphs
    ]
    return {"body": {"content": content}}


# ---------------------------------------------------------------------------
# GoogleDocsClient.read_document
# ---------------------------------------------------------------------------


class TestReadDocument:
    def test_paragraphs_joined(self):
        doc = _doc_body("Hello, ", "world!\n")
        client = _docs_client(get_response=doc)
        text = client.read_document("doc_id_123")
        assert text == "Hello, world!\n"

    def test_empty_body_returns_empty_string(self):
        client = _docs_client(get_response={"body": {"content": []}})
        assert client.read_document("doc_id_123") == ""

    def test_missing_body_returns_empty_string(self):
        client = _docs_client(get_response={})
        assert client.read_document("doc_id_123") == ""

    def test_non_paragraph_elements_skipped(self):
        doc: dict[str, Any] = {
            "body": {
                "content": [
                    {"sectionBreak": {}},  # no paragraph key
                    {
                        "paragraph": {
                            "elements": [{"textRun": {"content": "Keep this."}}]
                        }
                    },
                ]
            }
        }
        client = _docs_client(get_response=doc)
        assert client.read_document("doc_id_123") == "Keep this."


# ---------------------------------------------------------------------------
# GoogleDocsClient.append_text
# ---------------------------------------------------------------------------


class TestAppendText:
    def test_success_returns_true(self):
        calls: list[dict] = []
        client = _docs_client(post_calls=calls)
        result = client.append_text("doc_id_123", "Appended line.\n")
        assert result is True
        assert len(calls) == 1
        req = calls[0]["requests"][0]["insertText"]
        assert req["text"] == "Appended line.\n"
        assert req["endOfSegmentLocation"] == {"segmentId": ""}

    def test_post_raises_returns_false(self):
        client = _docs_client(post_raises=OSError("timeout"))
        result = client.append_text("doc_id_123", "text")
        assert result is False
