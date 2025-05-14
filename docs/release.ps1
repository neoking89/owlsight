param(
    [Parameter(Mandatory=$true)]
    [string]$Version  # Example usage: ./release.ps1 -Version 1.2.3
)

$BranchName = "release/v$Version"
$TagName = "v$Version"

function Fail($msg) {
    Write-Host "❌ ERROR: $msg" -ForegroundColor Red
    exit 1
}

# Step 0: Check if branch or tag exists
Write-Host "🔍 Checking if branch or tag already exists..."

# Check for existing branch
$existingBranch = git branch -r | Select-String "origin/$BranchName"
if ($existingBranch) {
    Fail "Branch '$BranchName' already exists on remote."
}

# Check for existing tag
$existingTag = git tag | Where-Object { $_ -eq $TagName }
if ($existingTag) {
    Fail "Git tag '$TagName' already exists."
}

# Create and switch to the release branch
Write-Host "🌿 Creating and checking out branch '$BranchName'..."
git checkout -b $BranchName
if ($LASTEXITCODE -ne 0) { Fail "Could not create or switch to branch." }

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
if ($LASTEXITCODE -ne 0) { Fail "Git commit failed." }

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

Write-Host "✅ Release $Version completed successfully!" -ForegroundColor Green
