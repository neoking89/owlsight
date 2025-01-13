# TODO: Update matrix so it works on Windows too. Make a seperate dockerfile first for Windows.

# Exit on any error
$ErrorActionPreference = "Stop"

# Define the matrix of Python versions and base images
$matrix = @(
    @{ BaseImage = "python:3.9-slim"; OS = "linux" }
    @{ BaseImage = "python:3.12-slim"; OS = "linux" }
    @{ BaseImage = "mcr.microsoft.com/windows/nanoserver:ltsc2025"; OS = "windows"; PythonVersion = "3.11" }
    # @{ BaseImage = "mcr.microsoft.com/windows/python:3.12"; OS = "windows" }
)

# Loop through all combinations in the matrix
foreach ($entry in $matrix) {
    $BaseImage = $entry.BaseImage
    $OS = $entry.OS

    Write-Host "=== Testing with base image $BaseImage on $OS ===" -ForegroundColor Cyan

    # Build the Docker image using dockerfile.test
    docker build --build-arg BASE_IMAGE=$BaseImage -t owlsight-test -f dockerfile.test .

    # Run the tests
    if ($OS -eq "linux") {
        docker run --rm owlsight-test
    } elseif ($OS -eq "windows") {
        # Ensure Docker is configured for Windows containers
        docker run --rm owlsight-test
    } else {
        Write-Host "Unknown OS type: $OS" -ForegroundColor Red
        exit 1
    }

    # Cleanup Docker artifacts for this iteration
    Write-Host "Cleaning up Docker artifacts for $BaseImage" -ForegroundColor Yellow
    try {
        docker rmi -f owlsight-test
        docker system prune -f --volumes
    } catch {
        Write-Host "Cleanup failed: $_" -ForegroundColor Red
    }

    Write-Host "=== Cleanup for $BaseImage on $OS completed ===" -ForegroundColor Green
}

# Final global cleanup
Write-Host "Performing final Docker cleanup..." -ForegroundColor Yellow
try {
    docker system prune -af --volumes
} catch {
    Write-Host "Final cleanup failed: $_" -ForegroundColor Red
}

Write-Host "All tests passed and cleanup completed!" -ForegroundColor Green
