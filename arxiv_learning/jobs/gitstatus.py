"""Log Status of Git Repository"""
import git


def get_repository_status():
    """Get a Dictionary with the Status of the Current Repo"""
    repo = git.Repo(".")
    commit = repo.head.commit
    diffs = []
    for diff in repo.head.commit.diff(None, create_patch=True):
        if diff.renamed_file:
            diffs.append({"file" : diff.a_path, "renamed" : True})
            continue
        if diff.a_path is not None:
            if diff.a_path.endswith(".py"):
                diffs.append({"file" : diff.a_path, "patch" : diff.diff.decode()})
    for f in repo.untracked_files:
        if f.endswith(".py"):
            diffs.append({"file":f, "patch":open(f, "r").read()})
    return {"commit" : commit.hexsha,
            "log" : commit.summary,
            "date" : commit.committed_datetime.isoformat(),
            "author" : str(commit.author),
            "diffs" : diffs}

if __name__ == '__main__':
    STATUS = get_repository_status()
    DIFFS = STATUS["diffs"]
    del STATUS["diffs"]
    import json
    print("Commit")
    print(json.dumps(STATUS, indent=4))
    print("Diffs")
    for i, d in enumerate(DIFFS):
        print("Patch {}: {}".format(i, d["file"]))
        print("\n".join(d["patch"].split("\n")[:10]))
