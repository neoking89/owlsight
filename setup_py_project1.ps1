# Initialize variables
$venvFolder = ".venv"
$vscodeFolder = ".vscode"
$backupFolder = ".vscode_backup"

# Function to roll back changes
function Rollback {
    Write-Host "Rolling back changes..."
    if (Test-Path $backupFolder) {
        Copy-Item -Path "$backupFolder\*" -Destination $vscodeFolder -Recurse -Force
        Remove-Item -Path $backupFolder -Recurse -Force
    }
    if (Test-Path $venvFolder) {
        Remove-Item -Path $venvFolder -Recurse -Force
    }
    exit 1
}

# Check for Python installation
$pythonExe = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $pythonExe) {
    Write-Host "No Python installation found."
    exit 1
}

# Show the Python version
& $pythonExe --version
if ($LASTEXITCODE -ne 0) { Rollback; exit 1; }

# Create a virtual environment
& $pythonExe -m venv $venvFolder
if ($LASTEXITCODE -ne 0) { Rollback; exit 1; }

# Activate the virtual environment
. ".\$venvFolder\Scripts\Activate"

# Backup existing .vscode folder, if it exists
if (Test-Path $vscodeFolder) {
    Copy-Item -Path $vscodeFolder -Destination $backupFolder -Recurse -Force
}

# Create .vscode directory and JSON files
New-Item -Path $vscodeFolder -ItemType Directory -Force
@'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File (myenv)",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {
                "PYDEVD_DISABLE_FILE_VALIDATION": "1",
                "PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT": "60",
                "PYDEVD_WARN_SLOW_EVALUATION_TIMEOUT": "60"
            },
            "justMyCode": false
        }
    ]
}
'@ | Set-Content -Path "$vscodeFolder\launch.json"

@'
{
    "python.terminal.activateEnvironment": true
}
'@ | Set-Content -Path "$vscodeFolder\settings.json"

# If we reach this point, everything was successful; remove the backup
if (Test-Path $backupFolder) {
    Remove-Item -Path $backupFolder -Recurse -Force
}

Write-Host "Setup completed successfully."
