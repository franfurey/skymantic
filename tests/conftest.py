# tests/conftest.py
import sys
from pathlib import Path

# Adjust the path so Python will find the modules under src
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

