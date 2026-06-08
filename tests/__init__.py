from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PKG = ROOT
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
