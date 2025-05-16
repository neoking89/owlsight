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
Write-Host "[INFO] Checking if branch or tag already exists..."

# Check remote branch
$branchExistsRemote = git ls-remote --heads origin $BranchName
if ($branchExistsRemote) {
    if (Confirm "Remote branch '$BranchName' already exists. Delete and recreate it?") {
        Write-Host "[INFO] Deleting remote branch '$BranchName'..."
        git push origin --delete $BranchName
        if ($LASTEXITCODE -ne 0) { Fail "Failed to delete remote branch '$BranchName'." }
    } else {
        Fail "Aborted due to existing remote branch."
    }
}

# Check local branch and switch/create
if (git rev-parse --verify $BranchName 2>$null) {
    git checkout $BranchName
    if ($LASTEXITCODE -ne 0) { Fail "Failed to switch to existing local branch." }
} else {
    git checkout -b $BranchName
    if ($LASTEXITCODE -ne 0) { Fail "Failed to create branch '$BranchName'." }
}

# Check tag
$existingTag = git tag | Where-Object { $_ -eq $TagName }
if ($existingTag) {
    if (Confirm "Tag '$TagName' already exists. Delete and recreate it?") {
        Write-Host "[INFO] Deleting local tag '$TagName'..."
        git tag -d $TagName
        if ($LASTEXITCODE -ne 0) { Fail "Failed to delete local tag." }

        Write-Host "[INFO] Deleting remote tag '$TagName'..."
        git push --delete origin $TagName
        if ($LASTEXITCODE -ne 0) { Fail "Failed to delete remote tag." }
    } else {
        Fail "Aborted due to existing tag."
    }
}

# Step 1: Run tests
Write-Host "[INFO] Running pytest..."
pytest -vvv
if ($LASTEXITCODE -ne 0) { Fail "Local pytest failed." }

# Write-Host "[INFO] Running Docker tests..."
# ./docker/run_tests.ps1
# if ($LASTEXITCODE -ne 0) { Fail "Docker tests failed." }

# Step 2: Update README
Write-Host "[INFO] Updating README.md..."
python src/owlsight/docs/readme.py
if ($LASTEXITCODE -ne 0) { Fail "README update script failed." }

# Step 3: Commit
Write-Host "[INFO] Committing changes..."
git add .
if ($LASTEXITCODE -ne 0) { Fail "Git add failed." }

git commit -m "Update version to $Version"
# Safe even if nothing changed:
if ($LASTEXITCODE -ne 0) { Write-Host "[INFO] No changes to commit." }

# Step 4: Tag
Write-Host "[INFO] Tagging release '$TagName'..."
git tag -a $TagName -m "New release version $Version"
if ($LASTEXITCODE -ne 0) { Fail "Tag creation failed." }

# Step 5: Push
Write-Host "[INFO] Pushing branch '$BranchName' and tag '$TagName'..."
git push origin $BranchName
if ($LASTEXITCODE -ne 0) { Fail "Failed to push branch." }

git push origin $TagName
if ($LASTEXITCODE -ne 0) { Fail "Failed to push tag." }

# Step 6: Build distribution
Write-Host "[INFO] Building package..."
python -m build
if ($LASTEXITCODE -ne 0) { Fail "Build failed." }


Write-Host "`n✅ Release $Version completed successfully!" -ForegroundColor Green
