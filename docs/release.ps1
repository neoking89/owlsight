param(
    [Parameter(Mandatory = $true)]
    [string]$Version  # Example: ./release.ps1 -Version 2.6.0
)

$BranchName = "release/v$Version"
$TagName = "v$Version"

function Fail($msg) {
    Write-Host "`n[ERROR] $msg`n" -ForegroundColor Red
    exit 1
}

function Confirm($message) {
    $response = Read-Host "$message [y/N]"
    return $response -match '^[Yy]'
}

# Step 0: Check if branch or tag exists
Write-Host "🔍 Checking if branch or tag already exists..."

$branchExistsRemote = git ls-remote --heads origin $BranchName
if ($branchExistsRemote) {
    if (Confirm "Branch '$BranchName' already exists on remote. Delete and recreate it?") {
        Write-Host "🗑️ Deleting remote branch '$BranchName'..."
        git push origin --delete $BranchName
        if ($LASTEXITCODE -ne 0) { Fail "Failed to delete remote branch '$BranchName'." }
    } else {
        Fail "Aborted due to existing remote branch."
    }
}

# Check for existing tag
$existingTag = git tag | Where-Object { $_ -eq $TagName }
if ($existingTag) {
    if (Confirm "Tag '$TagName' already exists. Delete and recreate it?") {
        Write-Host "🗑️ Deleting local and remote tag '$TagName'..."
        git tag -d $TagName
        if ($LASTEXITCODE -ne 0) { Fail "Failed to delete local tag." }

        git push --delete origin $TagName
        if ($LASTEXITCODE -ne 0) { Fail "Failed to delete remote tag." }

        Write-Host "✅ Tag '$TagName' deleted successfully."
    } else {
        Fail "Aborted due to existing tag."
    }
}

# Create and switch to the release branch (even if local exists)
if (git rev-parse --verify $BranchName 2>$null) {
    git checkout $BranchName
    if ($LASTEXITCODE -ne 0) { Fail "Failed to switch to existing local branch." }
} else {
    git checkout -b $BranchName
    if ($LASTEXITCODE -ne 0) { Fail "Could not create branch '$BranchName'." }
}

# Step 1: Run tests
Write-Host "🧪 Running tests locally with pytest..."
pytest -vvv
if ($LASTEXITCODE -ne 0) { Fail "Local pytest failed." }

Write-Host "🐳 Running cross-platform tests with Docker..."
./docker/run_tests.ps1
if ($LASTEXITCODE -ne 0) { Fail "Docker tests failed." }

# Step 2: Update README
Write-Host "📝 Updating README.md..."
python src/owlsight/docs/readme.py
if ($LASTEXITCODE -ne 0) { Fail "README update script failed." }

# Step 3: Commit changes
Write-Host "💾 Committing changes..."
git add .
if ($LASTEXITCODE -ne 0) { Fail "Git add failed." }

git commit -m "Update version to $Version"
if ($LASTEXITCODE -ne 0) { Write-Host "⚠️ No changes to commit." }

# Step 4: Create Git tag
Write-Host "🏷️ Tagging release with '$TagName'..."
git tag -a $TagName -m "New release version $Version"
if ($LASTEXITCODE -ne 0) { Fail "Tag creation failed." }

# Step 5: Push to remote
Write-Host "🚀 Pushing branch and tag to remote..."
git push origin $BranchName
if ($LASTEXITCODE -ne 0) { Fail "Failed to push branch." }

git push origin $TagName
if ($LASTEXITCODE -ne 0) { Fail "Failed to push tag." }

Write-Host "`n✅ Release $Version completed successfully!" -ForegroundColor Green
