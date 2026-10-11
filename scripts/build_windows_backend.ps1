$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$backendPath = Join-Path $repositoryRoot "backend"
$tauriPath = Join-Path $repositoryRoot "desktop\tauri\src-tauri"
$binaryDirectory = Join-Path $tauriPath "binaries"
$python = Get-Command python -ErrorAction Stop
$rustc = Join-Path $env:USERPROFILE ".cargo\bin\rustc.exe"
$buildEnvironment = Join-Path $env:TEMP "MarketVisionAI-build-venv"
$buildPython = Join-Path $buildEnvironment "Scripts\python.exe"
$binaryDirectory = Join-Path $tauriPath "binaries"

if (-not (Test-Path $rustc)) {
    throw "Rust is required. Install the stable x86_64-pc-windows-msvc toolchain first."
}

$targetTriple = (& $rustc --print host-tuple).Trim()
if ($LASTEXITCODE -ne 0 -or $targetTriple -ne "x86_64-pc-windows-msvc") {
    throw "This packaging script supports x64 Windows MSVC only; detected '$targetTriple'."
}

if (-not (Test-Path (Join-Path $buildEnvironment "pyvenv.cfg"))) {
    if (Test-Path $buildEnvironment) {
        Remove-Item -LiteralPath $buildEnvironment -Recurse -Force
    }
    & $python.Source -m venv $buildEnvironment
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the isolated Python packaging environment."
    }
}
if (-not (Test-Path $buildPython)) {
    throw "The isolated Python packaging environment is missing its interpreter."
}

Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& $buildPython -m pip install -r (Join-Path $backendPath "requirements-build.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the backend packaging dependencies."
}

New-Item -ItemType Directory -Force -Path $binaryDirectory | Out-Null
$tesseractHome = $env:TESSERACT_HOME
if (-not $tesseractHome) {
    $tesseractHome = Join-Path $env:ProgramFiles "Tesseract-OCR"
}
$tesseractExecutable = Join-Path $tesseractHome "tesseract.exe"
$tesseractData = Join-Path $tesseractHome "tessdata"
if (-not (Test-Path -LiteralPath $tesseractExecutable) -or
    -not (Test-Path -LiteralPath (Join-Path $tesseractData "eng.traineddata"))) {
    throw "Install the official Tesseract OCR runtime with English data, or set TESSERACT_HOME before packaging."
}

$tesseractResource = Join-Path $binaryDirectory "tesseract"
if (Test-Path -LiteralPath $tesseractResource) {
    Remove-Item -LiteralPath $tesseractResource -Recurse -Force
}
New-Item -ItemType Directory -Force -Path (Join-Path $tesseractResource "tessdata") | Out-Null
Copy-Item -LiteralPath $tesseractExecutable -Destination $tesseractResource
Get-ChildItem -LiteralPath $tesseractHome -Filter "*.dll" -File |
    Copy-Item -Destination $tesseractResource
Copy-Item -LiteralPath (Join-Path $tesseractData "eng.traineddata") `
    -Destination (Join-Path $tesseractResource "tessdata")
$tesseractTsvConfig = Join-Path $tesseractData "configs\tsv"
if (-not (Test-Path -LiteralPath $tesseractTsvConfig)) {
    throw "Tesseract TSV output configuration is missing from the selected runtime."
}
New-Item -ItemType Directory -Force -Path `
    (Join-Path $tesseractResource "tessdata\configs") | Out-Null
Copy-Item -LiteralPath $tesseractTsvConfig `
    -Destination (Join-Path $tesseractResource "tessdata\configs")
$tesseractLicense = Join-Path $tesseractHome "doc\LICENSE"
if (-not (Test-Path -LiteralPath $tesseractLicense)) {
    throw "Tesseract license file is missing from the selected runtime."
}
New-Item -ItemType Directory -Force -Path (Join-Path $tesseractResource "doc") | Out-Null
Copy-Item -LiteralPath $tesseractLicense -Destination (Join-Path $tesseractResource "doc")
$tesseractAuthors = Join-Path $tesseractHome "doc\AUTHORS"
if (Test-Path -LiteralPath $tesseractAuthors) {
    Copy-Item -LiteralPath $tesseractAuthors -Destination (Join-Path $tesseractResource "doc")
}
$tesseractCheck = & (Join-Path $tesseractResource "tesseract.exe") `
    --tessdata-dir (Join-Path $tesseractResource "tessdata") --list-langs 2>&1
$tesseractCheckText = $tesseractCheck -join "`n"
if ($LASTEXITCODE -ne 0 -or $tesseractCheckText -notmatch "(?m)^\s*eng\s*$") {
    throw "The copied Tesseract runtime did not pass its offline language-data check."
}

$distPath = Join-Path $env:TEMP "MarketVisionAI-PyInstaller\dist"
$workPath = Join-Path $env:TEMP "MarketVisionAI-PyInstaller\build"
$specPath = Join-Path $env:TEMP "MarketVisionAI-PyInstaller\spec"
$migrationData = (Join-Path $backendPath "database\migrations") + ";database\migrations"

Push-Location $backendPath
try {
    & $buildPython -m PyInstaller `
        --noconfirm `
        --clean `
        --noconsole `
        --onedir `
        --manifest (Join-Path $backendPath "windows-dpi.manifest") `
        --add-data $migrationData `
        --collect-all cv2 `
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
