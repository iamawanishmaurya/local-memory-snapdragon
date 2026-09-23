# build_vec0.ps1 - OPTIONAL vendored build of sqlite-vec's vec0.dll for
# Windows ARM64 (PERF-01, plan 06-01 Task 4).
#
# sqlite-vec 0.1.9 ships no win_arm64 wheel, so on Snapdragon hosts the
# extension must be compiled from the single-file `sqlite-vec.c` amalgamation
# with MSVC targeting ARM64. The resulting DLL is placed in
# local_memory/store/bin/ where local_memory/store/vec_ann.py finds it via
# conn.load_extension().
#
# Usage:
#   powershell -File scripts/build_vec0.ps1            # download amalgamation
#   powershell -File scripts/build_vec0.ps1 -Src C:\path\sqlite-vec.c
#   powershell -File scripts/build_vec0.ps1 -Cl C:\path\cl.exe
#
# This script is never run by tests or the server. If MSVC is absent it
# prints a soft-fail message and exits 0 - the brute-force numpy fallback
# stays active until the DLL is built.
param(
    [string]$OutDir = "local_memory/store/bin",
    [string]$Src = "",
    [string]$Cl = ""
)

$ErrorActionPreference = "Stop"

Write-Host "build_vec0: optional vendored sqlite-vec vec0.dll build (ARM64)"

# --- locate cl.exe via vswhere unless -Cl given ------------------------------
if (-not $Cl) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
        if ($vsRoot) {
            $cl = Get-ChildItem -Path "$vsRoot\VC\Tools\MSVC\*\bin\HostARM64\ARM64\cl.exe" -ErrorAction SilentlyContinue |
                  Select-Object -First 1
            if (-not $cl) {
                $cl = Get-ChildItem -Path "$vsRoot\VC\Tools\MSVC\*\bin\Hostx64\ARM64\cl.exe" -ErrorAction SilentlyContinue |
                      Select-Object -First 1
            }
            if ($cl) { $Cl = $cl.FullName }
        }
    }
}
if (-not $Cl -or -not (Test-Path $Cl)) {
    Write-Host "MSVC (cl.exe) not found - this build is OPTIONAL. Brute-force numpy fallback stays active."
    Write-Host "Install 'Visual Studio Build Tools' with the ARM64 C++ workload and re-run."
    exit 0
}
Write-Host "cl.exe: $Cl"

# --- obtain the amalgamation --------------------------------------------------
$tmp = Join-Path $env:TEMP ("sqlite-vec-build-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    if ($Src) {
        if (-not (Test-Path $Src)) { throw "staged amalgamation not found: $Src" }
        Copy-Item $Src (Join-Path $tmp "sqlite-vec.c")
    }
    else {
        $url = "https://github.com/asg017/sqlite-vec/raw/v0.1.9/src/sqlite-vec.c"
        Write-Host "Downloading $url"
        Invoke-WebRequest -Uri $url -OutFile (Join-Path $tmp "sqlite-vec.c") -UseBasicParsing
    }

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Push-Location $tmp
    try {
        & $Cl /O2 /arm64 sqlite-vec.c /link /dll /out:vec0-arm64.dll
        if ($LASTEXITCODE -ne 0) { throw "cl.exe failed with exit code $LASTEXITCODE" }
    }
    finally { Pop-Location }

    $dest = Join-Path $OutDir "vec0-arm64.dll"
    Copy-Item (Join-Path $tmp "vec0-arm64.dll") $dest -Force
    $abs = (Resolve-Path $dest).Path
    Write-Host "OK: vendored DLL written to $abs"
    Write-Host "The loader (local_memory/store/vec_ann.py) probes local_memory/store/bin/vec0-arm64.dll"
    Write-Host "automatically; python -m local_memory.main --doctor should now report: sqlite-vec: available (vendored)"
}
catch {
    Write-Host "build_vec0 failed (soft): $_"
    Write-Host "This build is OPTIONAL - brute-force fallback stays active."
    exit 0
}
finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
