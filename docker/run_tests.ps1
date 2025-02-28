

param(
[string]$PythonVersion = "3.10"
)
docker build --build-arg BASE_IMAGE=python:${PythonVersion}-slim -t owlsight-test -f docker/dockerfile.test.linux .


# # Set variables based on platform
# if ($Platform -eq "linux") {
#     $dockerfile = "dockerfile.test.linux"
#     $tag = "owlsight-test-linux"
#     $buildArgs = "--build-arg BASE_IMAGE=python:${PythonVersion}-slim"
# } else {
#     $dockerfile = "dockerfile.test.windows"
#     $tag = "owlsight-test-windows"
#     $buildArgs = ""
# }

# Write-Host "Running tests for $Platform with Python $PythonVersion..."

# try {
#     # Build and run sequence
#     $buildCmd = "docker build $buildArgs -t $tag -f docker/$dockerfile ."
#     $runCmd = "docker run --rm $tag"
#     $cleanupCmd = "docker rmi -f $tag"

#     # Execute commands
#     Invoke-Expression $buildCmd
#     if ($LASTEXITCODE -eq 0) {
#         Invoke-Expression $runCmd
#         Invoke-Expression $cleanupCmd
#     }
# } catch {
#     Write-Error "Error: $_"
#     exit 1
# }