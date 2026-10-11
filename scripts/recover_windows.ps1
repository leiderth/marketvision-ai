[CmdletBinding()]
param(
    [string]$ApplicationPath
)

$ErrorActionPreference = "Stop"
$healthUri = "http://127.0.0.1:18731/"
$appData = Join-Path $env:LOCALAPPDATA "MarketVision AI"
$logDirectory = Join-Path $appData "logs"

function Test-MarketVisionBackend {
    try {
        $response = Invoke-RestMethod -Uri $healthUri -TimeoutSec 2
        return $response.app -eq "MarketVision AI"
    }
    catch {
        return $false
    }
}

function Show-MarketVisionLogs {
    foreach ($name in @("crash.log", "backend.log")) {
        $path = Join-Path $logDirectory $name
        if (Test-Path -LiteralPath $path) {
            Write-Output "--- $path (últimas 60 líneas) ---"
            Get-Content -LiteralPath $path -Tail 60
        }
    }
}

if (Test-MarketVisionBackend) {
    Write-Output "MarketVision AI responde correctamente en el puerto local 18731."
    exit 0
}

$runningApp = Get-Process -Name "marketvision-ai" -ErrorAction SilentlyContinue
if ($runningApp) {
    Write-Warning "La interfaz sigue abierta pero la API local no responde. Cierra MarketVision AI desde su ventana y vuelve a ejecutar este script; no se finalizarán procesos automáticamente."
    Show-MarketVisionLogs
    exit 1
}

if (-not $ApplicationPath) {
    $uninstallRoots = @(
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*"
    )
    $installEntry = $null
    foreach ($root in $uninstallRoots) {
        $installEntry = Get-ItemProperty -Path $root -ErrorAction SilentlyContinue |
            Where-Object { $_.DisplayName -eq "MarketVision AI" } |
            Select-Object -First 1
        if ($installEntry) {
            break
        }
    }

    if ($installEntry) {
        if ($installEntry.InstallLocation) {
            $ApplicationPath = Join-Path $installEntry.InstallLocation "marketvision-ai.exe"
        }
        elseif ($installEntry.DisplayIcon) {
            $ApplicationPath = ($installEntry.DisplayIcon -split ",")[0].Trim('"')
        }
    }
}

if (-not $ApplicationPath -or -not (Test-Path -LiteralPath $ApplicationPath)) {
    Write-Warning "No se encontró el ejecutable instalado. Pasa su ruta con -ApplicationPath."
    Show-MarketVisionLogs
    exit 1
}

Write-Output "Iniciando MarketVision AI desde $ApplicationPath"
Start-Process -FilePath $ApplicationPath
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Seconds 1
    if (Test-MarketVisionBackend) {
        Write-Output "La interfaz y el backend local responden correctamente."
        exit 0
    }
}

Write-Warning "La API no respondió tras iniciar la interfaz. Revisa los logs locales."
Show-MarketVisionLogs
exit 1
