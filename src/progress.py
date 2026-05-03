"""
In-place two-section progress display.

Writes to stderr so it works cleanly when stdout is redirected to a log file.
Display format (5 lines, updated in-place):

Running main.py  (  42 / 100 )
Estimate: 8 minute(s), 12 second(s) left

Running step: [Step 9] Centrality — Thread  (  63 / 100 )
Estimate: 1 minute(s), 4 second(s)
"""

import shutil
import sys
import threading
import time

_N_LINES = 5  # total display lines (including blank separator)


def _fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if not h:
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    return ", ".join(parts) if parts else "0 seconds"


class ProgressTracker:
    """
    Tracks overall pipeline progress and current-step progress.

    Each step has:
        name     — shown in the "Running step:" line
        weight   — relative fraction of total work (used for overall %)
        estimate — expected wall-clock duration in seconds (used for step ETA)

    Overall % is based on completed weights + current step's time-based fraction.
    Step   % is based on elapsed / estimate, capped at 99 until start_step advances.
    """

    def __init__(self, steps: list[dict]) -> None:
        self.steps = steps
        self._total_w: float = sum(s["weight"] for s in steps)
        self._done_w: float = 0.0
        self._idx: int = -1
        self._step_start: float = 0.0
        self._t0: float = time.time()
        self._lock = threading.Lock()
        self._printed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def start_step(self, idx: int) -> None:
        """Advance to step idx and reset the per-step timer."""
        with self._lock:
            if 0 <= self._idx < len(self.steps):
                self._done_w += self.steps[self._idx]["weight"]
            self._idx = idx
            self._step_start = time.time()
        self._render()

    def finish(self) -> None:
        """Stop the refresh thread and move the cursor below the display."""
        self._stop.set()
        self._thread.join(timeout=2)
        if self._printed:
            sys.stderr.write("\n")
            sys.stderr.flush()

    # ── internal ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.wait(1.0):
            self._render()

    def _render(self) -> None:
        with self._lock:
            idx = self._idx
            if idx < 0 or idx >= len(self.steps):
                return
            step = self.steps[idx]
            elapsed_step = time.time() - self._step_start
            est = step["estimate"]
            step_frac = min(elapsed_step / est, 0.99) if est > 0 else 0.0

            done = self._done_w + step["weight"] * step_frac
            overall_pct = 100.0 * done / self._total_w if self._total_w else 0.0

            elapsed_total = time.time() - self._t0
            if overall_pct > 0.5:
                global_eta = elapsed_total * (100.0 - overall_pct) / overall_pct
            else:
                global_eta = float(sum(s["estimate"] for s in self.steps))

            step_eta = max(est - elapsed_step, 0.0)

        width = shutil.get_terminal_size(fallback=(120, 24)).columns - 1
        lines = [
            f"Running main.py  ( {overall_pct:3.0f} / 100 )"[:width],
            f"Estimate: {_fmt_eta(global_eta)} left"[:width],
            "",
            f"Running step: {step['name']}  ( {step_frac * 100:3.0f} / 100 )"[:width],
            f"Estimate: {_fmt_eta(step_eta)}"[:width],
        ]

        out = ""
        if self._printed:
            # move up (_N_LINES - 1) lines to reach the first line, then col 0
            out += f"\033[{_N_LINES - 1}A\r"

        for i, line in enumerate(lines):
            out += f"\r{line}\033[K"
            if i < len(lines) - 1:
                out += "\n"

        sys.stderr.write(out)
        sys.stderr.flush()
        self._printed = True
