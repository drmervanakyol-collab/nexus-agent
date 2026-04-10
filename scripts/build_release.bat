@echo off
setlocal enabledelayedexpansion
:: =============================================================================
:: scripts/build_release.bat
:: Nexus Agent — Full Release Build Pipeline
::
:: Steps:
::   1. Run full CI suite (make ci)  — aborts on any failure
::   2. Inject build metadata into environment
::   3. Download / verify Tesseract binary into tools/tesseract/
::   4. Run PyInstaller  → dist/nexus-agent/
::   5. Run Inno Setup   → dist/nexus-agent-vX.Y.Z-setup.exe
::
:: Required tools (must be on PATH or specified via env vars):
::   make, python, pyinstaller, iscc (Inno Setup compiler)
::
:: Optional env vars (override defaults):
::   NEXUS_VERSION       — semantic version (default: from nexus/release/version.py)
::   NEXUS_TESSERACT_SRC — path to local tesseract.exe to embed (skips download prompt)
::   NEXUS_TESSDATA_SRC  — path to local tessdata/ dir to embed
::   CI_PROFILE          — CI profile passed to make (default: release)
:: =============================================================================

:: ---------------------------------------------------------------------------
:: Banner
:: ---------------------------------------------------------------------------
echo.
echo =============================================================
echo  Nexus Agent — Release Build Pipeline
echo =============================================================
echo.

:: ---------------------------------------------------------------------------
:: Step 0 — Resolve root directory
:: ---------------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
pushd "%ROOT_DIR%"
set "ROOT_DIR=%CD%"
echo [0/5] Root: %ROOT_DIR%

:: ---------------------------------------------------------------------------
:: Step 1 — CI gate: all tests must pass
:: ---------------------------------------------------------------------------
echo.
echo [1/5] Running CI suite (profile=%CI_PROFILE%)...
if not defined CI_PROFILE set "CI_PROFILE=release"

make ci-%CI_PROFILE%
if errorlevel 1 (
    echo.
    echo [FAIL] CI suite failed. Aborting release build.
    popd
    exit /b 1
)
echo [OK]  CI suite passed.

:: ---------------------------------------------------------------------------
:: Step 2 — Inject build metadata
:: ---------------------------------------------------------------------------
echo.
echo [2/5] Resolving build metadata...

:: Version from version.py if not provided
if not defined NEXUS_VERSION (
    for /f "delims=" %%v in ('python -c "from nexus.release.version import VERSION; print(VERSION)"') do (
        set "NEXUS_VERSION=%%v"
    )
)

:: Build date (ISO-8601)
for /f "delims=" %%d in ('python -c "from datetime import date; print(date.today().isoformat())"') do (
    set "NEXUS_BUILD_DATE=%%d"
)

:: Git hash (first 8 chars of HEAD)
for /f "delims=" %%h in ('git rev-parse --short=8 HEAD 2^>nul') do (
    set "NEXUS_GIT_HASH=%%h"
)
if not defined NEXUS_GIT_HASH set "NEXUS_GIT_HASH=unknown"

echo     Version  : %NEXUS_VERSION%
echo     Date     : %NEXUS_BUILD_DATE%
echo     Git hash : %NEXUS_GIT_HASH%

:: Export for PyInstaller spec (reads os.environ at build time)
set "NEXUS_VERSION=%NEXUS_VERSION%"
set "NEXUS_BUILD_DATE=%NEXUS_BUILD_DATE%"
set "NEXUS_GIT_HASH=%NEXUS_GIT_HASH%"

:: ---------------------------------------------------------------------------
:: Step 3 — Tesseract binary
:: ---------------------------------------------------------------------------
echo.
echo [3/5] Preparing Tesseract binary...

set "TOOLS_TESS=%ROOT_DIR%\tools\tesseract"
if not exist "%TOOLS_TESS%" mkdir "%TOOLS_TESS%"
if not exist "%TOOLS_TESS%\tessdata" mkdir "%TOOLS_TESS%\tessdata"

if defined NEXUS_TESSERACT_SRC (
    if exist "%NEXUS_TESSERACT_SRC%" (
        copy /y "%NEXUS_TESSERACT_SRC%" "%TOOLS_TESS%\tesseract.exe" >nul
        echo     Copied tesseract.exe from %NEXUS_TESSERACT_SRC%
    ) else (
        echo [WARN] NEXUS_TESSERACT_SRC not found: %NEXUS_TESSERACT_SRC%
        echo        Tesseract will NOT be embedded — OCR requires system install.
    )
) else if exist "%TOOLS_TESS%\tesseract.exe" (
    echo     Using cached tesseract.exe in tools/tesseract/
) else (
    echo [WARN] No Tesseract binary found. Set NEXUS_TESSERACT_SRC to embed it.
    echo        OCR features will require a separate Tesseract installation.
)

if defined NEXUS_TESSDATA_SRC (
    if exist "%NEXUS_TESSDATA_SRC%" (
        xcopy /e /y /q "%NEXUS_TESSDATA_SRC%\*" "%TOOLS_TESS%\tessdata\" >nul
        echo     Copied tessdata/ from %NEXUS_TESSDATA_SRC%
    )
)

:: ---------------------------------------------------------------------------
:: Step 4 — PyInstaller
:: ---------------------------------------------------------------------------
echo.
echo [4/5] Running PyInstaller...

if exist "dist\nexus-agent" (
    echo     Removing old dist\nexus-agent\...
    rmdir /s /q "dist\nexus-agent"
)

pyinstaller nexus.spec --noconfirm
if errorlevel 1 (
    echo.
    echo [FAIL] PyInstaller failed.
    popd
    exit /b 1
)

:: Set NEXUS_TESSERACT_PATH inside the bundle for runtime health check
:: (The executable reads this env var to locate the embedded binary.)
echo     Setting NEXUS_TESSERACT_PATH in bundle...
set "BUNDLE_TESS=%ROOT_DIR%\dist\nexus-agent\tesseract\tesseract.exe"
if exist "%BUNDLE_TESS%" (
    echo     Tesseract embedded at: %BUNDLE_TESS%
) else (
    echo     [WARN] Tesseract not present in bundle — OCR path will fall back to PATH.
)

echo [OK]  PyInstaller build complete: dist\nexus-agent\

:: ---------------------------------------------------------------------------
:: Step 5 — Inno Setup
:: ---------------------------------------------------------------------------
echo.
echo [5/5] Building installer with Inno Setup...

:: Locate iscc
set "ISCC_PATH="
where iscc >nul 2>&1 && set "ISCC_PATH=iscc"
if not defined ISCC_PATH (
    if exist "C:\Program Files (x86)\Inno Setup 6\iscc.exe" (
        set "ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\iscc.exe"
    )
)
if not defined ISCC_PATH (
    if exist "C:\Program Files\Inno Setup 6\iscc.exe" (
        set "ISCC_PATH=C:\Program Files\Inno Setup 6\iscc.exe"
    )
)

if not defined ISCC_PATH (
    echo [FAIL] Inno Setup compiler (iscc.exe) not found.
    echo        Install from https://jrsoftware.org/isinfo.php or add to PATH.
    popd
    exit /b 1
)

:: Export env vars consumed by the .iss script
set "NEXUS_DIST_DIR=%ROOT_DIR%\dist\nexus-agent"
set "NEXUS_OUTPUT_DIR=%ROOT_DIR%\dist"

"%ISCC_PATH%" "installer\nexus_agent.iss"
if errorlevel 1 (
    echo.
    echo [FAIL] Inno Setup compilation failed.
    popd
    exit /b 1
)

:: Verify output
set "SETUP_EXE=%ROOT_DIR%\dist\nexus-agent-v%NEXUS_VERSION%-setup.exe"
if not exist "%SETUP_EXE%" (
    echo [FAIL] Expected installer not found: %SETUP_EXE%
    popd
    exit /b 1
)

:: ---------------------------------------------------------------------------
:: Done
:: ---------------------------------------------------------------------------
echo.
echo =============================================================
echo  BUILD SUCCESSFUL
echo  Installer: dist\nexus-agent-v%NEXUS_VERSION%-setup.exe
echo  Version  : %NEXUS_VERSION% (%NEXUS_BUILD_DATE%, %NEXUS_GIT_HASH%)
echo =============================================================
echo.

popd
endlocal
exit /b 0
