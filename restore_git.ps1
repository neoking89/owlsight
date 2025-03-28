# This script restores .git folder from remote repository

# Require the repository URL as a parameter
param(
    [Parameter(Mandatory=$true, HelpMessage="Enter the GitHub repository URL")]
    [string]$repoUrl
)

# Validate the repository URL format
if (-not ($repoUrl -match '^https://github.com/[^/]+/[^/]+\.git$')) {
    Write-Host "Error: Invalid GitHub repository URL format. Must be like 'https://github.com/username/repository.git'" -ForegroundColor Red
    exit 1
}

# Check if we're in a Git repository
if (-not (Test-Path .git)) {
    # Initialize a new Git repository with 'main' as the default branch
    git init -b main

    # Add the remote repository using the specified URL
    git remote add origin $repoUrl
    
    # Fetch all branches and history
    git fetch origin

    # Reset to the main branch
    git reset --hard origin/main

    # Restore all untracked files that might have been in the original repository
    git clean -fd

    Write-Host "Git repository successfully restored to its original state on 'main' branch." -ForegroundColor Green
} else {
    Write-Host "Error: .git folder already exists. No restoration needed." -ForegroundColor Red
    exit 1
}