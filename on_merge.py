import subprocess
import sys
from typing import Optional


def run_command(command: str) -> str:
    """
    Run a shell command and return its output. If the command fails, log the error and exit.
    """
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error running command: {command}", file=sys.stderr)
        print(f"Error details: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def git_operations(operation: str, branch: str="", main_branch: str= "main") -> None:
    """
    Perform Git operations like pull, switch branches, merge, and push.
    """
    operations = {
        "pull": f"Pulling the latest changes from {branch}...",
        "switch_main": f"Switching to the {main_branch} branch...",
        "pull_main": f"Pulling the latest changes from the {main_branch} branch...",
        "merge": f"Merging {branch} into {main_branch}...",
        "push": "Pushing changes to the remote repository...",
        "switch_back": f"Switching back to branch {branch}...",
    }
    commands = {
        "pull": f"git pull origin {branch}",
        "switch_main": f"git checkout {main_branch}",
        "pull_main": f"git pull origin {main_branch}",
        "merge": f"git merge --no-ff {branch}",
        "push": f"git push origin {main_branch}",
        "switch_back": f"git checkout {branch}",
    }

    if operation not in operations:
        print(f"Error: Unknown operation '{operation}'", file=sys.stderr)
        sys.exit(1)

    print(operations[operation])
    try:
        run_command(commands[operation])
    except KeyError:
        print(f"Error: No command defined for operation '{operation}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during '{operation}': {e}", file=sys.stderr)
        sys.exit(1)


def run_tests() -> None:
    """
    Run pytest to validate that tests pass before proceeding.
    """
    print("Running tests...")
    result = subprocess.run(["pytest", "-vvv"], text=True)
    if result.returncode != 0:
        print("Tests failed, aborting merge.", file=sys.stderr)
        sys.exit(1)
    print("Tests passed.")


def get_current_branch() -> str:
    """
    Get the current Git branch.
    """
    return run_command("git rev-parse --abbrev-ref HEAD")


def get_last_version() -> str:
    """
    Retrieve the last Git tag (version). If no tags exist, return a default version or handle an empty repo.
    """
    try:
        return run_command("git describe --tags --abbrev=0")
    except subprocess.CalledProcessError:
        # Check if repository has no commits
        commit_count = run_command("git rev-list --count HEAD")
        if commit_count == "0":
            return "0.1.0"  # Default for an empty repository
        else:
            print("Error: Could not determine the last version. Is the repository initialized?", file=sys.stderr)
            sys.exit(1)


def bump_version(version: str, level: str) -> str:
    """
    Increment the version number based on the specified bump level (major, minor, patch).
    """
    version_parts = list(map(int, version.lstrip("v").split(".")))
    if level == "major":
        version_parts[0] += 1
        version_parts[1] = version_parts[2] = 0
    elif level == "minor":
        version_parts[1] += 1
        version_parts[2] = 0
    else:  # patch
        version_parts[2] += 1
    return ".".join(map(str, version_parts))


def determine_version_bump(last_version: str) -> str:
    """
    Determine whether the version bump should be a major, minor, or patch based on commit messages.
    """
    commit_messages = run_command(f"git log {last_version}..HEAD --pretty=format:'%s'").splitlines()
    if any("BREAKING CHANGE" in msg for msg in commit_messages):
        return "major"
    if any(msg.startswith("feat") for msg in commit_messages):
        return "minor"
    return "patch"


def tag_and_push_version(new_version: str) -> None:
    """
    Tag the new version and push the tag to the remote repository.
    """
    print(f"Tagging the new version: v{new_version}")
    run_command(f"git tag v{new_version}")
    run_command("git push --tags")


def commit_version_bump(new_version: str) -> None:
    """
    Commit the version bump with an appropriate message and push the changes to the remote repository.
    """
    print(f"Committing the version bump: v{new_version}")
    run_command(f"git commit -am 'chore: bump version to v{new_version}'")
    git_operations("push")


def main(main_branch: Optional[str] = "main") -> None:
    """
    Main function to execute the full process: testing, merging, version bumping, and tagging.
    """
    # Run tests
    run_tests()

    # Get the current branch and ensure it's up-to-date
    current_branch = get_current_branch()
    git_operations("pull", current_branch, main_branch)

    # Switch to main branch, pull latest, merge the current branch
    git_operations("switch_main", "", main_branch)
    git_operations("pull_main", "", main_branch)
    git_operations("merge", current_branch, main_branch)
    git_operations("push", "", main_branch)

    # Determine the new version
    last_version = get_last_version()
    bump_level = determine_version_bump(last_version)
    new_version = bump_version(last_version, bump_level)

    # Tag and push the new version, commit the bump
    tag_and_push_version(new_version)
    commit_version_bump(new_version)

    # # Switch back to the original branch
    # git_operations("switch_back", current_branch)

    print(f"Main branch {main_branch} updated with latest changes from {current_branch} and version bumped to v{new_version}.")


if __name__ == "__main__":
    main()
