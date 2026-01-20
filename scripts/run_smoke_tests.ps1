# Standard smoke test invocations for options-iv-surface
# Usage: run from repo root: .\scripts\run_smoke_tests.ps1
# Note: CLI opens result HTMLs in your default browser on success.

$ErrorActionPreference = 'Stop'

$steps = @(
    @{ Name = 'Validate testdata01'; Command = 'python plot_vol_surface.py --validate -i testdata01.csv' },
    @{ Name = 'Surface chart'; Command = 'python plot_vol_surface.py -i testdata01.csv -o vol_analysis.html' },
    @{ Name = 'Line/pin chart'; Command = 'python plot_vol_surface.py -i testdata01.csv --mode line --target-dte 7 -o pin_analysis.html' },
    @{ Name = 'Diff chart'; Command = 'python plot_vol_surface.py -i testdata01.csv --diff testdata02.csv -o diff_analysis.html' }
)

foreach ($step in $steps) {
    Write-Host "[RUN] $($step.Name)" -ForegroundColor Cyan
    Invoke-Expression $step.Command
}

Write-Host "All smoke steps completed." -ForegroundColor Green
