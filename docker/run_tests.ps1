# Script to run Docker tests with multiple Python versions
# Loops through Python 3.9 to 3.12, builds the Docker image, runs it, saves logs, then cleans up

# Define the range of Python versions to test
$pythonVersions = @("3.9", "3.10", "3.11", "3.12")

# Create a logs directory if it doesn't exist
$logsDir = "test_logs"
if (-not (Test-Path -Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
    Write-Host "Created logs directory: $logsDir"
}

foreach ($version in $pythonVersions) {
    $imageName = "owlsight-test:python-$version"
    $buildLogFile = Join-Path -Path $logsDir -ChildPath "python-$version-build-log.txt"
    $testLogFile = Join-Path -Path $logsDir -ChildPath "python-$version-test-log.txt"
    
    Write-Host "========================================="
    Write-Host "Testing with Python version $version"
    Write-Host "========================================="

    # 1. Build the Docker image with the specified Python version
    Write-Host "Building Docker image with Python $version..."
    $buildOutput = docker build --build-arg BASE_IMAGE=python:${version}-slim -t $imageName -f docker/dockerfile.test.linux . 2>&1
    $buildOutput | Out-File -FilePath $buildLogFile -Encoding utf8
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to build image for Python $version. See log for details." -ForegroundColor Red
        "BUILD FAILED" | Out-File -FilePath $testLogFile
        continue
    }

    # 2. Run the Docker container to execute the tests
    Write-Host "Running tests with Python $version..."
    $testOutput = docker run --rm $imageName 2>&1
    $testOutput | Out-File -FilePath $testLogFile -Encoding utf8
    
    # Display a summary of the test results
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Tests for Python $version completed successfully." -ForegroundColor Green
    } else {
        Write-Host "Tests for Python $version failed with exit code $LASTEXITCODE." -ForegroundColor Red
    }
    
    # 3. Delete the created image
    Write-Host "Cleaning up: Removing Docker image for Python $version..."
    docker rmi $imageName
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully removed image $imageName."
    } else {
        Write-Host "Warning: Failed to remove image $imageName." -ForegroundColor Yellow
    }
    
    Write-Host "Completed testing with Python $version. Logs saved to: $buildLogFile and $testLogFile"
    Write-Host ""
}

Write-Host "All tests completed. Logs available in $logsDir directory."
