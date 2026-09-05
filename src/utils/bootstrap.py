"""Re-run an entry point under the project's ``.venv`` when launched elsewhere.

``python cli.py`` or ``python eval/eval_runner.py`` from a shell where the venv
is not activated picks up the global Python, which lacks the project's
dependencies. The failure is not a clean "No module named langchain_openai": it
surfaces inside a LangGraph task as "During task with name 'agent'", which the
eval runner then records as an ordinary case failure. Rather than fail late, hand
off to the venv interpreter if one exists.

Import this before anything that pulls in a third-party package. It only needs
the standard library and the ``src`` package itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def reexec_in_venv(script: str) -> None:
    """Restart ``script`` under ``.venv`` unless already running there.

    ``CALORAI_NO_REEXEC=1`` disables the hand-off (set automatically on the
    child so it cannot loop).
    """
    if os.environ.get("CALORAI_NO_REEXEC"):
        return
    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    ]
    venv_python = next((c for c in candidates if c.is_file()), None)
    if venv_python is None:
        return
    try:
        if Path(sys.executable).resolve() == venv_python.resolve():
            return
        if Path(sys.prefix).resolve() == (PROJECT_ROOT / ".venv").resolve():
            return
    except OSError:
        return
    os.environ["CALORAI_NO_REEXEC"] = "1"
    sys.stdout.flush()
    raise SystemExit(
        subprocess.call([str(venv_python), str(Path(script).resolve()), *sys.argv[1:]])
    )
