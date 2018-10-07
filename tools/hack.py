import os
import subprocess
from pathlib import Path

def source_import(fpath: Path, globals=globals(), locals=locals()):
    with fpath.open('r') as fin:
        exec(fin.read(), globals, locals)

def git_repo_root() -> Path:
    _raw_output = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip()
    return Path(_raw_output.decode('ascii'))
