param(
    [ValidateSet("setup", "test", "serve", "health", "help")]
    [string]$Action = "help",
    [ValidateRange(1, 65535)][int]$Port = 8000
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = if ($env:PROJECT_PYTHON) { $env:PROJECT_PYTHON } else {
    $WindowsPython = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
    if (Test-Path $WindowsPython) { $WindowsPython }
    else { Join-Path $ProjectRoot ".venv/bin/python" }
}
function Invoke-Python {
    & $PythonPath @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}
Push-Location $ProjectRoot
try {
    switch ($Action) {
        "setup" {
            if (-not (Test-Path $PythonPath)) {
                $BasePython = if ($env:PYTHON) { $env:PYTHON } else { "python" }
                & $BasePython -m venv .venv
                if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed" }
                $WindowsPython = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
                $PythonPath = if (Test-Path $WindowsPython) { $WindowsPython } else { Join-Path $ProjectRoot ".venv/bin/python" }
            }
            Invoke-Python -m pip install --only-binary=:all: -r requirements-dev.txt
        }
        "test" { Invoke-Python -m unittest discover -s tests -v }
        "serve" { Invoke-Python -m uvicorn api:app --host 127.0.0.1 --port $Port }
        "health" {
            $Response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
            if ($Response.status -ne "ok") { throw "Health check failed" }
            Write-Output "Health check passed"
        }
        "help" { Write-Output "Usage: ./scripts/dev.ps1 -Action setup|test|serve|health [-Port 8000]" }
    }
} finally { Pop-Location }
