<#
.SYNOPSIS
  One-command demo: webcam or video -> MediaPipe -> gaze zone + eye/mouth signals
  -> causal sliding windows -> driver state overlay.

.EXAMPLE
  .\run_demo.ps1                          # webcam, calibrates gaze first if no profile exists
  .\run_demo.ps1 -Calibrate               # redo the 80 s guided gaze calibration
  .\run_demo.ps1 -NoGaze                  # eyes / yawn / head pose only
  .\run_demo.ps1 -Source clip.mp4 -Save out.mp4 -NoDisplay
#>
param(
    [string]$Source = "0",
    [switch]$Calibrate,
    [string]$Profile = "weights/gaze_profile.npz",
    [switch]$NoGaze,
    [switch]$NoDisplay,
    [string]$Save = "",
    [string[]]$Extra = @()
)
# MediaPipe writes harmless warnings to stderr; PowerShell 5.1 would treat them as errors.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONPATH = "src;scripts"
$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

$required = @("weights/face_landmarker_v2_with_blendshapes.task", "weights/blazegaze_mpiifacegaze.keras")
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) {
    Write-Host "Missing model files (see README, 'Weights'):" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" }
    exit 1
}

$isWebcam = $Source -match '^\d+$'
$haveAdapter = Test-Path "weights/master_4driver_blazegaze_dmd_adapter.pt"
if (-not $NoGaze -and ($Calibrate -or (-not (Test-Path $Profile) -and -not $haveAdapter))) {
    if (-not $isWebcam) {
        Write-Host "Gaze calibration needs a webcam (-Source 0); use -NoGaze or pass an existing profile." -ForegroundColor Red
        exit 1
    }
    Write-Host "Gaze calibration: look at each target as prompted (about 80 s). Press q to abort." -ForegroundColor Cyan
    & $python scripts/calibrate_gaze.py --source $Source --profile-out $Profile
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$demoArgs = @("scripts/demo_realtime.py", "--source", $Source)
if ($NoGaze) { $demoArgs += "--no-gaze" }
elseif (Test-Path $Profile) { $demoArgs += @("--gaze-profile", $Profile) }
if ($NoDisplay) { $demoArgs += "--no-display" }
if ($Save) { $demoArgs += @("--save", $Save) }
Write-Host "Running demo (q quit, space pause, r reset)" -ForegroundColor Cyan
& $python @demoArgs @Extra
exit $LASTEXITCODE
