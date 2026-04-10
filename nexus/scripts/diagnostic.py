"""
nexus/scripts/diagnostic.py
CLI entry point for Nexus Agent diagnostic export.

Usage
-----
  python -m nexus.scripts.diagnostic [OUTPUT_PATH]

  OUTPUT_PATH defaults to ``nexus_diagnostic.zip`` in the current directory.

The script collects all available diagnostic data from the running system
(health checks, system info) and writes a ZIP archive to OUTPUT_PATH.
Real log / task / transport data requires a live database; those collectors
default to empty lists when no DB is configured via NEXUS_STORAGE__DB_PATH.
"""
from __future__ import annotations

import json
import sys

from nexus.core.settings import NexusSettings
from nexus.infra.diagnostic import DiagnosticReporter


def _settings_dict(settings: NexusSettings) -> dict[str, object]:
    return json.loads(settings.model_dump_json())  # type: ignore[no-any-return]


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    output_path = args[0] if args else "nexus_diagnostic.zip"

    settings = NexusSettings()

    reporter = DiagnosticReporter(
        _get_settings_fn=lambda: _settings_dict(settings),
    )

    try:
        reporter.export_zip(output_path)
        print(f"Tanı dosyası oluşturuldu: {output_path}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
