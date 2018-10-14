import os
import subprocess
import sys
from pathlib import Path

def add_to_path(module_fpath: Path):
    if not module_fpath.exists():
        raise ValueError("cannot find {}".format(module_fpath))
    module_path = str(module_fpath)
    if module_path in sys.path:
        return
    sys.path.append(module_path)

def source_import(module_fpath: Path, globals=globals(), locals=locals()):
    with module_fpath.open('r') as fin:
        exec(fin.read(), globals, locals)

def git_repo_root() -> Path:
    _raw_output = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip()
    return Path(_raw_output.decode('ascii'))
