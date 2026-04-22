import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parents[1]
BASE_CODE_ROOT = REPO_ROOT / "base" / "code"

for path in (CODE_ROOT, BASE_CODE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
