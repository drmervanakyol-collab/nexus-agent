@echo off
setlocal enabledelayedexpansion
:: =============================================================================
:: scripts/sign_release.bat
:: Nexus Agent — Code Signing Pipeline
::
:: Signs and verifies:
::   1. nexus-agent.exe  (main binary)
::   2. installer.exe    (*-setup.exe in dist/)
::
:: Required tools:
::   signtool.exe  — Windows SDK (auto-discovered or via NEXUS_SIGNTOOL)
::   python        — for version lookup
::
:: Optional env vars:
::   NEXUS_SIGN_CERT     — cert thumbprint or subject name (store cert)
::   NEXUS_SIGN_PFX      — path to PFX file (dev / CI scenario)
::   NEXUS_SIGN_PFX_PASS — PFX password
::   NEXUS_SIGNTOOL      — full path to signtool.exe
::   NEXUS_VERSION       — override version (default: from version.py)
::   NEXUS_DIST_DIR      — dist directory (default: <repo-root>\dist)
:: =============================================================================

:: ---------------------------------------------------------------------------
:: Banner
:: ---------------------------------------------------------------------------
echo.
echo =============================================================
echo  Nexus Agent — Code Signing Pipeline
echo =============================================================
echo.

:: ---------------------------------------------------------------------------
:: Resolve root directory
:: ---------------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
pushd "%ROOT_DIR%"
set "ROOT_DIR=%CD%"
echo [0] Root: %ROOT_DIR%

:: ---------------------------------------------------------------------------
:: Resolve dist directory
:: ---------------------------------------------------------------------------
if not defined NEXUS_DIST_DIR set "NEXUS_DIST_DIR=%ROOT_DIR%\dist"
echo [0] Dist: %NEXUS_DIST_DIR%

:: ---------------------------------------------------------------------------
:: Locate signtool.exe
:: ---------------------------------------------------------------------------
echo.
echo [1/4] Locating signtool.exe...

set "SIGNTOOL="

if defined NEXUS_SIGNTOOL (
    if exist "%NEXUS_SIGNTOOL%" (
        set "SIGNTOOL=%NEXUS_SIGNTOOL%"
        echo     Using NEXUS_SIGNTOOL: %SIGNTOOL%
        goto :signtool_found
    )
)

where signtool >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%p in ('where signtool') do (
        set "SIGNTOOL=%%p"
        goto :signtool_found
    )
)

:: Search Windows SDK locations
for /d %%v in ("C:\Program Files (x86)\Windows Kits\10\bin\*") do (
    if exist "%%v\x64\signtool.exe" (
        set "SIGNTOOL=%%v\x64\signtool.exe"
        goto :signtool_found
    )
)

for %%f in (
    "C:\Program Files (x86)\Windows Kits\10\bin\x64\signtool.exe"
    "C:\Program Files (x86)\Windows Kits\10\bin\x86\signtool.exe"
    "C:\Program Files\Windows Kits\10\bin\x64\signtool.exe"
) do (
    if exist %%f (
        set "SIGNTOOL=%%~f"
        goto :signtool_found
    )
)

echo [FAIL] signtool.exe not found.
echo        Install Windows SDK or set NEXUS_SIGNTOOL to its full path.
popd
exit /b 1

:signtool_found
echo [OK]  signtool: %SIGNTOOL%

:: ---------------------------------------------------------------------------
:: Build signing flags
:: ---------------------------------------------------------------------------
set "SIGN_FLAGS=/fd SHA256 /tr http://timestamp.digicert.com /td SHA256"

if defined NEXUS_SIGN_PFX (
    if exist "%NEXUS_SIGN_PFX%" (
        set "SIGN_FLAGS=%SIGN_FLAGS% /f "%NEXUS_SIGN_PFX%""
        if defined NEXUS_SIGN_PFX_PASS (
            set "SIGN_FLAGS=%SIGN_FLAGS% /p "%NEXUS_SIGN_PFX_PASS%""
        )
        echo [1/4] Mode: PFX file (%NEXUS_SIGN_PFX%)
        goto :flags_ready
    ) else (
        echo [WARN] NEXUS_SIGN_PFX not found: %NEXUS_SIGN_PFX%
    )
)

if defined NEXUS_SIGN_CERT (
    set "SIGN_FLAGS=%SIGN_FLAGS% /n "%NEXUS_SIGN_CERT%""
    echo [1/4] Mode: Store cert (%NEXUS_SIGN_CERT%)
    goto :flags_ready
)

set "SIGN_FLAGS=%SIGN_FLAGS% /a"
echo [1/4] Mode: Auto (first available cert — dev/self-signed)

:flags_ready

:: ---------------------------------------------------------------------------
:: Resolve target binaries
:: ---------------------------------------------------------------------------
echo.
echo [2/4] Resolving target binaries...

set "AGENT_EXE=%NEXUS_DIST_DIR%\nexus-agent.exe"

:: Find installer: dist\*-setup.exe
set "INSTALLER_EXE="
for %%f in ("%NEXUS_DIST_DIR%\*-setup.exe") do (
    set "INSTALLER_EXE=%%f"
)

set "FOUND_ANY=0"

if exist "%AGENT_EXE%" (
    echo     nexus-agent.exe  : %AGENT_EXE%
    set "FOUND_ANY=1"
) else (
    echo [WARN] nexus-agent.exe not found: %AGENT_EXE%
)

if defined INSTALLER_EXE (
    echo     installer.exe    : %INSTALLER_EXE%
    set "FOUND_ANY=1"
) else (
    echo [WARN] No *-setup.exe found in %NEXUS_DIST_DIR%
)

if "%FOUND_ANY%"=="0" (
    echo [FAIL] No binaries found to sign. Run build_release.bat first.
    popd
    exit /b 1
)

:: ---------------------------------------------------------------------------
:: Sign + verify helper macro (implemented via goto)
:: ---------------------------------------------------------------------------
:: Usage: set TARGET=<path> then call :sign_and_verify

:: ---------------------------------------------------------------------------
:: Step 3 — Sign nexus-agent.exe
:: ---------------------------------------------------------------------------
if exist "%AGENT_EXE%" (
    echo.
    echo [3/4] Signing nexus-agent.exe...
    "%SIGNTOOL%" sign %SIGN_FLAGS% "%AGENT_EXE%"
    if errorlevel 1 (
        echo [FAIL] Signing failed: nexus-agent.exe
        popd
        exit /b 1
    )
    echo [OK]  Signed: nexus-agent.exe

    echo       Verifying nexus-agent.exe...
    "%SIGNTOOL%" verify /pa /v "%AGENT_EXE%"
    if errorlevel 1 (
        echo [FAIL] Verification failed: nexus-agent.exe
        popd
        exit /b 1
    )
    echo [OK]  Verified: nexus-agent.exe
) else (
    echo [3/4] Skipped nexus-agent.exe ^(not found^).
)

:: ---------------------------------------------------------------------------
:: Step 4 — Sign installer
:: ---------------------------------------------------------------------------
if defined INSTALLER_EXE (
    echo.
    echo [4/4] Signing installer...
    "%SIGNTOOL%" sign %SIGN_FLAGS% "%INSTALLER_EXE%"
    if errorlevel 1 (
        echo [FAIL] Signing failed: %INSTALLER_EXE%
        popd
        exit /b 1
    )
    echo [OK]  Signed: %INSTALLER_EXE%

    echo       Verifying installer...
    "%SIGNTOOL%" verify /pa /v "%INSTALLER_EXE%"
    if errorlevel 1 (
        echo [FAIL] Verification failed: %INSTALLER_EXE%
        popd
        exit /b 1
    )
    echo [OK]  Verified: %INSTALLER_EXE%
) else (
    echo [4/4] Skipped installer ^(not found^).
)

:: ---------------------------------------------------------------------------
:: Done
:: ---------------------------------------------------------------------------
echo.
echo =============================================================
echo  SIGNING SUCCESSFUL
echo =============================================================
echo.

popd
endlocal
exit /b 0
