import os
import sys
import shutil
from fnmatch import fnmatch
import argparse
import subprocess
from huggingface_hub import HfApi, Repository

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Upload files to a Hugging Face repository",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="Repository ID on Hugging Face (format: username/repo_name)",
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        help="Local directory path containing the files to upload",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="new-branch",
        help="Name of the branch to push to",
    )
    parser.add_argument(
        "--ignore_patterns",
        type=str,
        default="global_step*",
        help="Comma-separated list of patterns to ignore",
    )
    return parser.parse_args(args)


def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    api = HfApi()
    # Ensure the repository exists; create if it does not
    # No need for splitting username and repository name; directly pass the full repo_id
    api.create_repo(repo_id=args.repo_id, private=True, exist_ok=True)

    local_dir = f"{SCRIPT_DIR}/../../../hf-push-dir/temp_repo_{args.repo_id.replace('/', '_')}"
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)

    # # Clone the repo
    repo = Repository(local_dir=local_dir, clone_from=args.repo_id)
    # repo.git_checkout(branch=args.branch, create_branch_ok=True)
    repo.git_pull()  # optional, in case you want to sync first

    # Checkout to branch (create if it doesn’t exist)
    try:
        # Try to checkout the branch if it already exists remotely
        subprocess.run(["git", "checkout", args.branch], cwd=local_dir, check=True)
        subprocess.run(["git", "pull", "origin", args.branch], cwd=local_dir, check=True)
    except subprocess.CalledProcessError:
        # If it fails, create the branch
        subprocess.run(["git", "checkout", "-b", args.branch], cwd=local_dir, check=True)

    # Copy files from folder_path into the repo
    for root, _, files in os.walk(args.folder_path):
        for file in files:
            rel_dir = os.path.relpath(root, args.folder_path)
            rel_file = os.path.join(rel_dir, file) if rel_dir != "." else file

            if any(fnmatch(rel_file, pattern.strip()) for pattern in args.ignore_patterns.split(",")):
                continue

            src = os.path.join(root, file)
            dest = os.path.join(local_dir, rel_file)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)

    # repo.push_to_hub(commit_message="Upload to new branch", branch=args.branch)
    # Commit manually
    repo.git_add(auto_lfs_track=True)
    repo.git_commit("Upload to new branch")
    subprocess.run(["git", "push", "--set-upstream", "origin", args.branch], cwd=local_dir, check=True)

    print(f"Uploaded to {args.repo_id} on branch {args.branch}")


    # kwargs = {}
    # if args.ignore_patterns is not None:
    #     kwargs["ignore_patterns"] = args.ignore_patterns.split(",")

    # # Upload all contents from the specified local directory to the repository
    # api.upload_folder(
    #     folder_path=args.folder_path,
    #     repo_id=args.repo_id,
    #     repo_type="model",
    #     **kwargs,
    #     # Change to "space" if you're uploading to a Space
    # )
    # print(f"All files have been uploaded to {args.repo_id}")


if __name__ == "__main__":
    main()
