; nexus_agent.iss
; Inno Setup 6+ script for Nexus Agent Windows installer.
;
; Usage:
;   iscc installer\nexus_agent.iss
;
; Environment variables consumed (injected by build_release.bat):
;   NEXUS_VERSION      — e.g. "1.0.0"
;   NEXUS_DIST_DIR     — path to dist\nexus-agent\ (PyInstaller output)
;   NEXUS_OUTPUT_DIR   — destination directory for setup.exe (default: dist\)
;
; Prerequisites (place in installer\ before building):
;   installer\tesseract-setup.exe  — Tesseract OCR 5.x 64-bit silent installer
;     Download: https://github.com/UB-Mannheim/tesseract/wiki
;     Filename: tesseract-ocr-w64-setup-5.5.0.20241111.exe (rename to tesseract-setup.exe)
;
; Output: nexus-agent-v{NEXUS_VERSION}-setup.exe

#define AppName      "Nexus Agent"
#define AppPublisher "Nexus Agent Project"
#define AppURL       "https://github.com/nexus-agent/nexus-agent"
#define AppExeName   "nexus-agent.exe"
#define AppVersion   GetEnv("NEXUS_VERSION")
#define DistDir      GetEnv("NEXUS_DIST_DIR")
#define OutputDir    GetEnv("NEXUS_OUTPUT_DIR")

; Fallback defaults for local manual runs
#if AppVersion == ""
  #define AppVersion "1.0.0"
#endif
#if DistDir == ""
  #define DistDir "..\dist\nexus-agent"
#endif
#if OutputDir == ""
  #define OutputDir "..\dist"
#endif

[Setup]
; Unique GUID — must not change between versions (enables upgrade detection)
AppId={{A3F2B8C1-7E4D-4A9B-B6F5-2D1E8C0A3F7B}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
; Output
OutputDir={#OutputDir}
OutputBaseFilename=nexus-agent-v{#AppVersion}-setup
; Compression
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
; Appearance
WizardStyle=modern
; Privileges — install to user profile if not admin
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Architecture — 64-bit only
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
; Minimum OS version — Windows 10 (10.0)
MinVersion=10.0
; Uninstall info
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\{#AppExeName}
; Restart not required
RestartIfNeededByRun=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "startmenuicon"; Description: "Create Start Menu shortcut"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce

[Files]
; Main application bundle — everything from the PyInstaller dist directory
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Tesseract OCR installer — bundled, extracted to temp dir and run silently
Source: "tesseract-setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
; Desktop shortcut
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon; Comment: "Launch Nexus Agent"
; Start Menu group
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Comment: "Launch Nexus Agent"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"

[Run]
; Install Tesseract OCR silently (only if not already installed)
Filename: "{tmp}\tesseract-setup.exe"; \
  Parameters: "/VERYSILENT /NORESTART /COMPONENTS=""languages\eng,languages\tur"" /SP-"; \
  StatusMsg: "Installing Tesseract OCR..."; \
  Check: not IsTesseractInstalled; \
  Flags: waituntilterminated
; Optional: launch after install
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up runtime-generated files on uninstall
Type: filesandordirs; Name: "{localappdata}\NexusAgent\logs"
Type: filesandordirs; Name: "{localappdata}\NexusAgent\cache"

[Code]
// ---------------------------------------------------------------------------
// Tesseract detection — skip install if already present
// ---------------------------------------------------------------------------
function IsTesseractInstalled(): Boolean;
var
  sPath: String;
begin
  // Check registry for Tesseract uninstall key (both HKLM and HKCU)
  Result := RegQueryStringValue(HKEY_LOCAL_MACHINE,
      'SOFTWARE\Tesseract-OCR', 'InstallDir', sPath)
    or RegQueryStringValue(HKEY_CURRENT_USER,
      'SOFTWARE\Tesseract-OCR', 'InstallDir', sPath);
end;

// ---------------------------------------------------------------------------
// Upgrade detection — remove old version before installing new one
// ---------------------------------------------------------------------------
function InitializeSetup(): Boolean;
var
  sUninstaller: String;
  iResultCode: Integer;
begin
  Result := True;
  if RegQueryStringValue(
      HKEY_CURRENT_USER,
      'Software\Microsoft\Windows\CurrentVersion\Uninstall\{A3F2B8C1-7E4D-4A9B-B6F5-2D1E8C0A3F7B}_is1',
      'UninstallString',
      sUninstaller) then
  begin
    if MsgBox(
        'An existing installation of Nexus Agent was found. It will be uninstalled before continuing.',
        mbInformation, MB_OKCANCEL) = IDOK then
    begin
      Exec(RemoveQuotes(sUninstaller), '/SILENT /NORESTART', '', SW_HIDE, ewWaitUntilTerminated, iResultCode);
    end else begin
      Result := False;
    end;
  end;
end;
