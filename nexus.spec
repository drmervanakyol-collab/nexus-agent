# nexus.spec
# PyInstaller build specification for Nexus Agent.
#
# Build command (run from repo root):
#   pyinstaller nexus.spec
#
# Output: dist/nexus-agent/  (single-directory mode)
#
# Environment variables consumed at build time:
#   NEXUS_VERSION   — semantic version injected into version.py
#   NEXUS_BUILD_DATE
#   NEXUS_GIT_HASH
# ---------------------------------------------------------------------------

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(SPECPATH)                       # repo root
NEXUS_PKG = ROOT / "nexus"
CONFIGS_DIR = ROOT / "configs"
DOCS_DIR = ROOT / "docs"

# Tesseract bundled binary (expected in tools/tesseract/ by build script)
TESS_DIR = ROOT / "tools" / "tesseract"
TESS_EXE = TESS_DIR / "tesseract.exe"
TESSDATA_DIR = TESS_DIR / "tessdata"

# ---------------------------------------------------------------------------
# Hidden imports — runtime-loaded modules that the static analyser misses
# ---------------------------------------------------------------------------

hidden_imports = [
    # Anthropic SDK internals
    "anthropic",
    "anthropic._models",
    "anthropic.types",
    # OpenAI SDK internals
    "openai",
    "openai.types",
    # dxcam — ctypes-based screen capture
    "dxcam",
    "dxcam._libs",
    # pytesseract
    "pytesseract",
    # pywin32 / comtypes for UIA
    "win32api",
    "win32con",
    "win32gui",
    "pywintypes",
    "comtypes",
    "comtypes.client",
    # structlog / rich console
    "structlog",
    "rich",
    "rich.console",
    "rich.text",
    # numpy / Pillow (image processing)
    "numpy",
    "PIL",
    "PIL.Image",
    # tomllib / tomli fallback
    "tomllib",
    # Nexus release metadata
    "nexus.release.version",
    # Nexus integrations loaded by plugin manager
    "nexus.integrations",
]

# ---------------------------------------------------------------------------
# Data files (configs, docs, Tesseract binary + tessdata)
# ---------------------------------------------------------------------------

datas = [
    # Application configs bundled as read-only defaults
    (str(CONFIGS_DIR), "configs"),
    # Legal documents shown in onboarding
    (str(DOCS_DIR / "privacy_policy.md"), "docs"),
    (str(DOCS_DIR / "terms_of_service.md"), "docs"),
]

# Bundle Tesseract only when the tools/ directory was populated by build_release.bat
if TESS_EXE.is_file():
    datas.append((str(TESS_EXE), "tesseract"))

if TESSDATA_DIR.is_dir():
    datas.append((str(TESSDATA_DIR), os.path.join("tesseract", "tessdata")))

# ---------------------------------------------------------------------------
# Binaries (DLLs / PYDs that need explicit inclusion)
# ---------------------------------------------------------------------------

binaries = []

# dxcam ships C-extension DLLs — collect them alongside the package
try:
    import dxcam as _dxcam_mod
    _dxcam_path = Path(_dxcam_mod.__file__).parent
    for _dll in _dxcam_path.glob("*.dll"):
        binaries.append((str(_dll), "dxcam"))
    for _pyd in _dxcam_path.glob("*.pyd"):
        binaries.append((str(_pyd), "dxcam"))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

a = Analysis(
    ["nexus/__main__.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Dev / test dependencies — not needed at runtime
        "pytest",
        "hypothesis",
        "mypy",
        "ruff",
        "bandit",
        "safety",
        "vulture",
        "pyinstaller",
        # Jupyter / notebook stuff
        "IPython",
        "jupyter",
        "notebook",
        # Matplotlib (not used at runtime)
        "matplotlib",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ---------------------------------------------------------------------------
# EXE (the launcher inside the directory bundle)
# ---------------------------------------------------------------------------

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # single-directory mode
    name="nexus-agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,            # CLI application — keep console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=None,
    icon=None,
)

# ---------------------------------------------------------------------------
# COLLECT (single-directory bundle)
# ---------------------------------------------------------------------------

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="nexus-agent",      # dist/nexus-agent/
)
