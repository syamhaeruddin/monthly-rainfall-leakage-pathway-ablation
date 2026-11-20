# Waits for stages 1-2 to finish, then runs the surrogate test (stage 3).
Set-Location $PSScriptRoot
while (-not (Test-Path logs\run_all_done.txt)) { Start-Sleep 60 }
$py = "$HOME\.venv_methodsx\Scripts\python.exe"
& $py lpa_surrogate.py --fast 100 --step 5 --n_jobs 14 > logs\surrogate_stdout.log 2> logs\surrogate_stderr.log
"DONE $(Get-Date)" | Out-File logs\surrogate_done.txt
