import sys
from pathlib import Path

# Allow test modules to import from src/ without requiring PYTHONPATH=src.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
