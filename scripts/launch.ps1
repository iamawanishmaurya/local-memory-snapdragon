# Local Memory one-command launcher (DEMO-01 / D-01).
# Idempotent: safe to run twice. Never pip installs at runtime (fail_fast only).
#
# Bypass invocation:
#   powershell -ExecutionPolicy Bypass -File scripts/launch.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/launch.ps1 -Port 8787
param([int]$Port = 8787)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$Root = Split-Path $PSScriptRoot -Parent
$Py = "$Root/.venv-arm64/Scripts/python.exe"

# --- Venv gate: recreate when missing, direct-path invocation only ---
if (-not (Test-Path $Py)) {
    Write-Host "[launch] venv missing - creating .venv-arm64 ..."
    Push-Location $Root
    try {
        try {
            & py -3.12-arm64 -m venv .venv-arm64
        } catch {
            & python -m venv .venv-arm64
        }
    } finally {
        Pop-Location
    }
    $Py = "$Root/.venv-arm64/Scripts/python.exe"
    if (-not (Test-Path $Py)) {
        Write-Host "[launch] failed to create venv at $Py"
        exit 1
    }
    Write-Host "[launch] venv created."
} else {
    Write-Host "[launch] venv present - skipping creation."
}

# --- Deps gate: fail_fast only, never pip install ---
& $Py -c "from local_memory.deps import fail_fast; fail_fast()"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Fix: pip install -r requirements.txt"
    exit 1
}

# --- UI gate BEFORE server start: build only when dist missing ---
if (-not (Test-Path "$Root/ui/dist/index.html")) {
    Write-Host "[launch] ui/dist missing - building UI ..."
    $proc = Start-Process -FilePath "$Root/scripts/build_ui.bat" -WorkingDirectory $Root -Wait -PassThru -NoNewWindow
    if ($proc.ExitCode -ne 0) {
        Write-Host "[launch] UI build failed (exit $($proc.ExitCode))"
        exit 1
    }
} else {
    Write-Host "[launch] UI dist present - skipping npm build."
}

# --- Models gate: 5 base sentinels, each existing AND length over 100KB ---
$Sentinels = @(
    "nomic-embed-text.onnx",
    "nomic-tokenizer.json",
    "clip-vit-b32-image.onnx",
    "clip-vit-b32-text.onnx",
    "bpe_simple_vocab_16e6.txt.gz"
)
$ModelsMissing = $false
foreach ($name in $Sentinels) {
    $p = Join-Path "$Root/models" $name
    if (-not (Test-Path $p)) { $ModelsMissing = $true; break }
    if ((Get-Item $p).Length -le 100KB) { $ModelsMissing = $true; break }
}
if ($ModelsMissing) {
    Write-Host "[launch] base models missing - running scripts/setup_models.py (base only) ..."
    & $Py "$Root/scripts/setup_models.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[launch] model provisioning failed (exit $LASTEXITCODE)"
        exit 1
    }
} else {
    Write-Host "[launch] base models present - skipping download."
}

# --- Port probe: .NET TcpClient with 500ms timeout ---
$AlreadyRunning = $false
try {
    $client = New-Object System.Net.Sockets.TcpClient
    $iar = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
    $waited = $iar.AsyncWaitHandle.WaitOne(500)
    if ($waited -and $client.Connected) {
        $AlreadyRunning = $true
    }
    try { $client.Close() } catch {}
} catch {
    $AlreadyRunning = $false
}

if ($AlreadyRunning) {
    Write-Host "[launch] server already running at http://127.0.0.1:$Port/ - skipping start."
} else {
    Write-Host "[launch] starting server on port $Port ..."
    Start-Process -FilePath $Py -ArgumentList "-m local_memory.main --port $Port" -WorkingDirectory $Root
}

# --- Poll GET /api/status up to 30s, then open browser ---
$deadline = (Get-Date).AddSeconds(30)
$up = $false
while ((Get-Date) -lt $deadline) {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/status" -UseBasicParsing -TimeoutSec 3
        if ($resp.StatusCode -eq 200) { $up = $true; break }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
if (-not $up) {
    Write-Host "[launch] server did not answer GET /api/status within 30s"
    exit 1
}
Write-Host "[launch] server up at http://127.0.0.1:$Port/ - opening browser."
Start-Process "http://127.0.0.1:$Port/"
exit 0
