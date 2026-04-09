"""
nexus/integrations/google/docs.py
Google Docs API v1 client for Nexus Agent.

GoogleDocsClient
----------------
Thin, fully-injectable wrapper around the Docs REST API.

read_document(document_id) -> str
  GET docs/v1/documents/{id} and extract plain text by concatenating
  all paragraph text runs in body content order.

append_text(document_id, text) -> bool
  POST docs/v1/documents/{id}:batchUpdate with an insertText request
  at the end of the document body.
  Returns True on success, False on failure.

Injectable callables
--------------------
_get_fn  : (url, headers) -> dict
_post_fn : (url, headers, body) -> dict
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.request import Request, urlopen

from nexus.infra.logger import get_logger

_log = get_logger(__name__)

_DOCS_BASE = "https://docs.googleapis.com/v1/documents"


# ---------------------------------------------------------------------------
# Default injectable implementations
# ---------------------------------------------------------------------------


def _default_get(url: str, headers: dict[str, str]) -> dict[str, Any]:
    import urllib.request  # noqa: PLC0415

    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _default_post(
    url: str, headers: dict[str, str], body: dict[str, Any]
) -> dict[str, Any]:
    import urllib.request  # noqa: PLC0415

    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# GoogleDocsClient
# ---------------------------------------------------------------------------


class GoogleDocsClient:
    """
    Google Docs API v1 client.

    Parameters
    ----------
    get_token_fn:
        Callable returning a valid OAuth 2.0 access token string.
    _get_fn:
        ``(url, headers) -> dict``.  HTTP GET implementation.
    _post_fn:
        ``(url, headers, body) -> dict``.  HTTP POST implementation.
    """

    def __init__(
        self,
        get_token_fn: Callable[[], str],
        *,
        _get_fn: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
        _post_fn: (
            Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]] | None
        ) = None,
    ) -> None:
        self._get_token = get_token_fn
        self._get = _get_fn or _default_get
        self._post = _post_fn or _default_post

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_document(self, document_id: str) -> str:
        """
        Return the plain-text content of a Google Doc.

        Concatenates all paragraph text runs from the document body in
        order, preserving newlines between paragraphs.

        Parameters
        ----------
        document_id:
            The document ID (from its URL).

        Returns
        -------
        Plain-text string.  Empty string when the document has no body
        content.
        """
        url = f"{_DOCS_BASE}/{document_id}"
        doc = self._get(url, self._auth_headers())
        return self._extract_text(doc)

    def append_text(self, document_id: str, text: str) -> bool:
        """
        Append *text* at the end of the document body.

        Uses ``insertText`` at index 1 with ``endOfSegmentLocation`` so
        the text is inserted after all existing content.

        Parameters
        ----------
        document_id:
            The document ID.
        text:
            Plain text to append (newlines are preserved).

        Returns
        -------
        True on success, False on any failure.
        """
        url = f"{_DOCS_BASE}/{document_id}:batchUpdate"
        body: dict[str, Any] = {
            "requests": [
                {
                    "insertText": {
                        "text": text,
                        "endOfSegmentLocation": {"segmentId": ""},
                    }
                }
            ]
        }
        try:
            self._post(url, self._auth_headers(), body)
            _log.debug("append_text_ok", document_id=document_id, length=len(text))
            return True
        except Exception as exc:
            _log.warning("append_text_failed", document_id=document_id, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        token = self._get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_text(doc: dict[str, Any]) -> str:
        """
        Walk the Docs API body content and collect all text runs.

        Paragraph elements are separated by newlines; structural elements
        (tables, section breaks) are skipped.
        """
        body = doc.get("body", {})
        content: list[dict[str, Any]] = body.get("content", [])
        parts: list[str] = []
        for element in content:
            paragraph = element.get("paragraph")
            if paragraph is None:
                continue
            para_text = "".join(
                pe.get("textRun", {}).get("content", "")
                for pe in paragraph.get("elements", [])
            )
            parts.append(para_text)
        return "".join(parts)
