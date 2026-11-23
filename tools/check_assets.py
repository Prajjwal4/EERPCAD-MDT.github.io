"""Fail loudly on any missing asset. Never substitutes anything."""
import json, sys
from pathlib import Path
spec = json.load(open("project.json"))
missing = [a for a in spec["assets_required"] if not (Path("assets") / a).is_file()]
print("MISSING:" if missing else "All assets present.", *missing, sep="\n  " if missing else "")
sys.exit(1 if missing else 0)
