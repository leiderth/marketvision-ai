$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$backendPath = Join-Path $repositoryRoot "backend"
$tauriPath = Join-Path $repositoryRoot "desktop\tauri\src-tauri"
$binaryDirectory = Join-Path $tauriPath "binaries"
$python = Get-Command python -ErrorAction Stop
$rustc = Join-Path $env:USERPROFILE ".cargo\bin\rustc.exe"
$buildEnvironment = Join-Path $env:TEMP "MarketVisionAI-build-venv"
$buildPython = Join-Path $buildEnvironment "Scripts\python.exe"

if (-not (Test-Path $rustc)) {
    throw "Rust is required. Install the stable x86_64-pc-windows-msvc toolchain first."
}

$targetTriple = (& $rustc --print host-tuple).Trim()
if ($LASTEXITCODE -ne 0 -or $targetTriple -ne "x86_64-pc-windows-msvc") {
    throw "This packaging script supports x64 Windows MSVC only; detected '$targetTriple'."
}

if (-not (Test-Path $buildPython)) {
    & $python.Source -m venv $buildEnvironment
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the isolated Python packaging environment."
    }
}

Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& $buildPython -m pip install -r (Join-Path $backendPath "requirements-build.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the backend packaging dependencies."
}

New-Item -ItemType Directory -Force -Path $binaryDirectory | Out-Null
$distPath = Join-Path $env:TEMP "MarketVisionAI-PyInstaller\dist"
$workPath = Join-Path $env:TEMP "MarketVisionAI-PyInstaller\build"
$specPath = Join-Path $env:TEMP "MarketVisionAI-PyInstaller\spec"

Push-Location $backendPath
try {
    & $buildPython -m PyInstaller `
        --noconfirm `
        --clean `
        --onedir `
        --name marketvision-backend `
        --distpath $distPath `
        --workpath $workPath `
        --specpath $specPath `
        --collect-all uvicorn `
        (Join-Path $backendPath "desktop_server.py")
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed to build the local API sidecar."
    }
}
finally {
    Pop-Location
}

$resourceDirectory = Join-Path $binaryDirectory "marketvision-backend"
if (Test-Path $resourceDirectory) {
    Remove-Item -LiteralPath $resourceDirectory -Recurse -Force
}
Copy-Item -Recurse -Force (Join-Path $distPath "marketvision-backend") $resourceDirectory

& $python.Source (Join-Path $PSScriptRoot "generate_windows_icon.py")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to generate the Windows application icon."
}

Write-Output "Prepared packaged backend resources: $resourceDirectory"
