# Reruns the LPA audit with the LSTM meta-learner (stages 2 and 3). Output: results\lstm\, results\surrogate_lstm\.
Set-Location $PSScriptRoot
$py = "$HOME\.venv_methodsx\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
"START $(Get-Date)" | Out-File logs\run_lstm.txt
& $py lpa_main_lstm.py > logs\main_lstm_stdout.log 2> logs\main_lstm_stderr.log
"main exit $LASTEXITCODE $(Get-Date)" | Out-File logs\run_lstm.txt -Append
& $py lpa_surrogate_lstm.py --fast 100 --step 5 --n_jobs 14 > logs\surr_lstm_stdout.log 2> logs\surr_lstm_stderr.log
"surrogate exit $LASTEXITCODE $(Get-Date)" | Out-File logs\run_lstm.txt -Append
"DONE $(Get-Date)" | Out-File logs\run_lstm.txt -Append
