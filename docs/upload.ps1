# Load .env file
$envFilePath = ".env"
if (Test-Path $envFilePath) {
    Get-Content $envFilePath | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]*)\s*=\s*(.*)\s*$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim('"').Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value)
        }
    }
} else {
    Write-Error ".env file not found"
    exit 1
}

# Check API_TOKEN is set
if (-not $env:API_TOKEN) {
    Write-Error "API_TOKEN not set in .env file"
    exit 1
}

# Run twine commands
Write-Host "Checking dist/* with twine..."
twine check dist/*

Write-Host "Uploading package to PyPI..."
twine upload -u __token__ -p $env:API_TOKEN dist/* --verbose
