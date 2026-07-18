"""Enable `python -m radiologist_agent`.

Usage:
    python -m radiologist_agent                 # run the bundled CTPA demo case
    python -m radiologist_agent --interactive   # choose report-tree options + dictate
    python -m radiologist_agent --case path.json
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
