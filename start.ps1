$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Missing .venv. Create a Python environment and install requirements.txt first.'
}
if (-not $env:HF_HOME) {
    $env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface'
}
& $projectPython -m streamlit run (Join-Path $PSScriptRoot 'app.py') --server.address localhost @args
exit $LASTEXITCODE
