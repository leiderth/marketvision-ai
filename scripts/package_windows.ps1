$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$frontendPath = Join-Path $repositoryRoot "frontend"
$tauriPath = Join-Path $repositoryRoot "desktop\tauri"
$tauriCli = Join-Path $frontendPath "node_modules\@tauri-apps\cli\tauri.js"
$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"

if (-not (Test-Path $tauriCli)) {
    throw "Install the frontend dependencies first by running npm install in frontend."
}
if (-not (Test-Path (Join-Path $cargoBin "cargo.exe"))) {
    throw "Rust/Cargo is required. Install the stable x86_64-pc-windows-msvc toolchain first."
}
if (-not (Test-Path $vswhere)) {
    throw "Visual Studio C++ Build Tools are required to compile the Windows desktop app."
}

$vsInstallPath = (& $vswhere -latest -products "*" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if ($LASTEXITCODE -ne 0 -or -not $vsInstallPath) {
    throw "The Visual Studio C++ x64 tools were not found."
}
$vsDevCmd = Join-Path $vsInstallPath "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $vsDevCmd)) {
    throw "Visual Studio developer environment script is missing: $vsDevCmd"
}

$env:PATH = "$cargoBin;$env:PATH"
& (Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe") `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File (Join-Path $PSScriptRoot "build_windows_backend.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to prepare the packaged FastAPI backend."
}

$buildCommand = 'call "{0}" -arch=x64 >nul && set "PATH={1};%PATH%" && cd /d "{2}" && node "{3}" build --bundles nsis' -f `
    $vsDevCmd, $cargoBin, $tauriPath, $tauriCli
& $env:ComSpec /d /s /c $buildCommand
if ($LASTEXITCODE -ne 0) {
    throw "Tauri failed to build the Windows NSIS installer."
}
