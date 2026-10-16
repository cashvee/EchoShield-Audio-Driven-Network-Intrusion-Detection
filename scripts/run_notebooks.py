"""Execute all notebooks in place and save cell outputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def main() -> None:
    for nb in NOTEBOOKS:
        print(f"Executing {nb.name}...", flush=True)
        subprocess.run(
            [
                str(PYTHON), "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=7200",
                "--inplace",
                str(nb),
            ],
            cwd=ROOT,
            check=True,
        )
    print("All notebooks executed.", flush=True)


if __name__ == "__main__":
    main()
