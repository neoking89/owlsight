# Save this file as run-docker-matrix.ps1
# Ensure you have Docker installed and that Docker commands are available.

$ErrorActionPreference = 'Stop'

# Configure your matrix here
$pythonVersions = @('3.8', '3.9', '3.10')
$failedTests = @()

Write-Host "Starting Docker matrix tests..."

foreach ($version in $pythonVersions) {
    $safePythonVersion = $version.Replace('.', '')
    Write-Host "`nTesting Python $version..."
    
    try {
        # Build the Docker image for this Python version
        Write-Host "Building Docker image for Python $version..."
        docker build --target python$safePythonVersion -t test-python$safePythonVersion .

        if ($LASTEXITCODE -ne 0) {
            throw "Docker build failed"
        }

        # Run tests
        Write-Host "Running tests..."
        docker run --rm test-python$safePythonVersion pytest tests/

        if ($LASTEXITCODE -eq 0) {
            Write-Host "Tests passed for Python $version"
        }
        else {
            Write-Host "Tests failed for Python $version"
            $failedTests += $version
        }
    }
    catch {
        Write-Host "Error testing Python $($version): $($_)"
        $failedTests += $version
    }
    finally {
        # Clean up
        Write-Host "Cleaning up Docker resources..."
        docker rmi test-python$safePythonVersion -f 2>$null
    }
}

# Print summary
Write-Host "`nTest Summary:"
if ($failedTests.Count -gt 0) {
    Write-Host "Failed versions:"
    $failedTests | ForEach-Object {
        Write-Host "  - Python $_"
    }
    exit 1
}
else {
    Write-Host "All tests passed successfully!"
    exit 0
}
