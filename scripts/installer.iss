; Inno Setup Script for OBEY API Gateway
; Requires Inno Setup 6+ (https://jrsoftware.org/isinfo.php)
;
; Usage:
;   1. Run  scripts/build-release.ps1  first to populate ./release
;   2. Compile this script:  iscc scripts/installer.iss

#define MyAppName      "OBEY API Gateway"
#define MyAppVersion   "0.4.1"
#define MyAppPublisher "OBEY"
#define MyAppExeName   "ai-gateway.exe"
#define MyAppURL       "https://github.com/fdanobey/OBEY-api-gateway"

[Setup]
AppId={{B8A3F2E1-7C4D-4E5A-9F6B-1D2E3F4A5B6C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\release
OutputBaseFilename=OBEY-API-Gateway-Setup-{#MyAppVersion}
SetupIconFile=..\Assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
LicenseFile=
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startupentry"; Description: "Start {#MyAppName} on Windows login"; GroupDescription: "Startup:"; Flags: unchecked

[Files]
; Main executable (icon already embedded via winres)
Source: "..\release\ai-gateway.exe";       DestDir: "{app}"; Flags: ignoreversion

; Configuration files
Source: "..\release\config.yaml";          DestDir: "{app}"; Flags: ignoreversion onlyifdoesntexist uninsneveruninstall
Source: "..\release\config.example.yaml";  DestDir: "{app}"; Flags: ignoreversion

; Assets (icon + logo for tray/splash)
Source: "..\release\Assets\icon.ico";      DestDir: "{app}\Assets"; Flags: ignoreversion
Source: "..\release\Assets\logo.jpg";      DestDir: "{app}\Assets"; Flags: ignoreversion

; Documentation
Source: "..\release\README.md";            DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}";                Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\Assets\icon.ico"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}";      Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}";          Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\Assets\icon.ico"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{userstartup}\{#MyAppName}";          Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\Assets\icon.ico"; WorkingDir: "{app}"; Tasks: startupentry

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{app}\logs.db"
Type: files; Name: "{app}\logs.db-journal"
