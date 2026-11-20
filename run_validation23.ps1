param([int]$NJobs = 14)
# Validation 2 and 3: VMD cache, then the two stations x two L4 settings in parallel, then the summary.
# Requires results\dec_<station>.npz (lpa_decompose.py). Output: results\validation23\
Set-Location $PSScriptRoot
$py = "$HOME\.venv_methodsx\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path logs | Out-Null
"START $(Get-Date)" | Out-File logs\run_validation23.txt
& $py lpa_validation23.py vmd --n_jobs $NJobs > logs\val23_vmd_stdout.log 2> logs\val23_vmd_stderr.log
"vmd exit $LASTEXITCODE $(Get-Date)" | Out-File logs\run_validation23.txt -Append
$procs = @()
foreach ($st in @("Darwin", "Perth")) {
    foreach ($l4 in @("open", "closed")) {
        $procs += Start-Process -FilePath $py -ArgumentList @("lpa_validation23.py", "run", "--station", $st, "--l4", $l4) `
            -RedirectStandardOutput "logs\val23_${st}_${l4}_stdout.log" -RedirectStandardError "logs\val23_${st}_${l4}_stderr.log" `
            -WindowStyle Hidden -PassThru
    }
}
$procs | Wait-Process
foreach ($p in $procs) { "run exit $($p.ExitCode)" | Out-File logs\run_validation23.txt -Append }
& $py lpa_validation23.py summary > logs\val23_summary_stdout.log 2> logs\val23_summary_stderr.log
"summary exit $LASTEXITCODE $(Get-Date)" | Out-File logs\run_validation23.txt -Append
"DONE $(Get-Date)" | Out-File logs\run_validation23.txt -Append
