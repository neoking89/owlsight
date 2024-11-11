# Run tests first
$testResult = pytest
if ($LASTEXITCODE -ne 0) {
    Write-Error "Tests failed! Aborting merge process."
    exit 1
}

# Save the current branch name
$currentBranch = git rev-parse --abbrev-ref HEAD

# Ensure the current branch is up-to-date
Write-Output "Pulling latest changes from $currentBranch..."
git pull origin $currentBranch
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to pull latest changes from $currentBranch. Aborting merge."
    exit 1
}

# Switch to the main branch
Write-Output "Switching to main branch..."
git checkout main
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to switch to main branch. Aborting merge."
    exit 1
}

# Ensure the main branch is up-to-date
Write-Output "Pulling latest changes from main..."
git pull origin main
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to pull latest changes from main. Aborting merge."
    git checkout $currentBranch
    exit 1
}

# Merge the current branch into main
Write-Output "Merging $currentBranch into main..."
git merge --no-ff $currentBranch
if ($LASTEXITCODE -ne 0) {
    Write-Error "Merge conflict detected! Aborting merge."
    git merge --abort
    git checkout $currentBranch
    exit 1
}

# Push the changes to the remote repository
Write-Output "Pushing changes to remote repository..."
git push origin main
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to push changes to remote repository."
    git checkout $currentBranch
    exit 1
}

# Switch back to the original branch
# Write-Output "Switching back to $currentBranch..."
# git checkout $currentBranch
# if ($LASTEXITCODE -ne 0) {
#     Write-Error "Failed to switch back to $currentBranch."
#     exit 1
# }

Write-Output "Success! Main branch has been updated with the latest changes from $currentBranch."