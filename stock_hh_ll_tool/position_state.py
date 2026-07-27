import json
from pathlib import Path

STATE_PATH = Path(__file__).resolve().parent / "state" / "positions.json"

# Per-symbol state machine: absent/"FLAT" = no position (recomputed fresh
# every run, nothing to persist). "PENDING_ENTRY" = a signal fired on a
# previous run; today's run fills it at TODAY's actual open (realistic
# next-bar fill, not the signal bar's own close — a real gap in the
# original design that the stress-test agent flagged). "LONG" = an open
# position with a recorded entry price/date/stop.


def load_positions(path: Path = STATE_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_positions(positions: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(positions, indent=2, default=str))
