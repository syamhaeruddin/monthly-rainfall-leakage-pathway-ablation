# Re-runs the companion cross-station study [31] on Darwin and Perth only, in a copy and a fresh environment.
#   .\run_companion_2stations.ps1 -Src <companion project folder> -Dst <new folder>
param([Parameter(Mandatory)][string]$Src, [Parameter(Mandatory)][string]$Dst)
New-Item -ItemType Directory -Force $Dst | Out-Null
# folder names below are those of the companion project version that was run
robocopy $Src $Dst /E /XD hasil laporan .git .venv /NFL /NDL /NJH /NJS | Out-Null
New-Item -ItemType Directory -Force "$Dst\hasil", "$Dst\laporan\naskah", "$Dst\laporan\eksperimen", "$Dst\log" | Out-Null
$base = (Get-Command python).Source
& $base "$PSScriptRoot\patch_copy.py" $Dst
if ($LASTEXITCODE -ne 0) { throw "patch failed" }
& $base -m venv "$Dst\.venv"
$py = "$Dst\.venv\Scripts\python.exe"
& $py -m pip install -r "$Dst\requirements.txt" *> "$Dst\log\pip.log"
Set-Location $Dst
$env:STATIONS = "Darwin,Perth"; $env:WORKERS = "7"; $env:PYTHONIOENCODING = "utf-8"
& $py src\siapkan_data.py
& $py src\indeks_iklim.py
& $py src\run_all.py all
& $py "$PSScriptRoot\summarise.py" $Dst "$PSScriptRoot\companion_2stations_summary.json"
