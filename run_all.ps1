# Runs stage 1 (CEEMDAN cache) and then stage 2 (MethodsX experiments).
Set-Location $PSScriptRoot
$py = "$HOME\.venv_methodsx\Scripts\python.exe"
& $py lpa_decompose.py --n_jobs 14 > logs\decompose_stdout.log 2> logs\decompose_stderr.log
& $py lpa_main.py > logs\main_stdout.log 2> logs\main_stderr.log
"DONE $(Get-Date)" | Out-File logs\run_all_done.txt
