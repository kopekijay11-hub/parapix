import subprocess
import os

def test_hygiene():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    files = [f for f in out.stdout.splitlines() if f and f != "tests/test_repo_hygiene.py"]
    banned_strings = ["api_key", "password", "secret"]
    banned_paths = ["/home/", "/mnt/", "C:\\"]
    for f in files:
        if f.startswith("results/"):
            continue
        if f.endswith(".csv"):
            raise AssertionError(f"tracked csv: {f}")
        full = os.path.join(os.path.dirname(os.path.dirname(__file__)), f)
        try:
            with open(full, "rb") as fh:
                data = fh.read().decode("utf-8", errors="ignore")
        except Exception:
            continue
        for s in banned_strings:
            assert s not in data, f"{s} in {f}"
        for p in banned_paths:
            assert p not in data, f"{p} in {f}"
