# Resolve project venv Python for worker scripts.
$script:ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$script:VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error @"
.venv not found at: $VenvPython

Create and install deps:
  cd `"$ProjectRoot`"
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
"@
}

function Invoke-VenvPython {
    # Do NOT name parameter 'Args' — conflicts with PowerShell and opens bare REPL.
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PyArgs)
    & "$VenvPython" @PyArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
