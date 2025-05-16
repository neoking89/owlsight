# Get current Git branch name
$currentBranch = git rev-parse --abbrev-ref HEAD

# Check if current branch starts with "release/"
if (-not $currentBranch.StartsWith("release/")) {
    Write-Error "Current branch '$currentBranch' is not a release branch (release/vX.X.X)"
    exit 1
}

Write-Host "Merging $currentBranch into main..."

# Checkout main
git checkout main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Merge release branch
git merge $currentBranch
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Push main to remote
git push origin main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Successfully merged $currentBranch into main and pushed."
