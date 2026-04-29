
#define MyAppName "STGNN Recommendation Desktop"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "ST-GNN Thesis Project"
#define MyAppExeName "STGNN_Desktop.exe"

[Setup]
AppId={{9A6B18D4-EA77-4F1B-B1E1-2F5B6A7C0D11}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\STGNNDesktop
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=D:\GNN_Thesis\installer_out
OutputBaseFilename=STGNN_Desktop_Setup_v10_REDESIGN
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
SetupIconFile=

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional tasks:"; Flags: unchecked

[Files]
Source: "D:\GNN_Thesis\dist\STGNN_Desktop\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
