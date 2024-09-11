# run tests first
pytest

# Save the current branch name
$currentBranch = git rev-parse --abbrev-ref HEAD

# Ensure the current branch is up-to-date
git pull origin $currentBranch

# Switch to the main branch
git checkout main

# Ensure the main branch is up-to-date
git pull origin main

# Merge the current branch into main
git merge --no-ff $currentBranch

# Push the changes to the remote repository
git push origin main

# # Switch back to the original branch
# git checkout $currentBranch

Write-Output "Main branch has been updated with the latest changes from $currentBranch."
