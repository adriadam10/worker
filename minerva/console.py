import collections
import threading
import time

import humanize
from rich import box
from rich.console import Console, Group
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from minerva.constants import HISTORY_LINES

console = Console()

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class WorkerDisplay:
    """
    Terminal display:
      • recent completed/failed jobs (oldest scrolls off, hidden when empty)
      • divider rule
      • one row per active worker slot
      • session stats footer
    """

    def __init__(self) -> None:
        self.history: collections.deque = collections.deque(maxlen=HISTORY_LINES)
        self.active: dict = {}  # file_id -> dict
        self._lock = threading.Lock()
        self._session_start = time.monotonic()
        self._total_done = 0
        self._total_bytes = 0

    def job_start(self, file_id: int, label: str) -> None:
        now = time.monotonic()
        with self._lock:
            self.active[file_id] = dict(
                label=label,
                status="DL",
                size=0,
                done=0,
                start_time=now,
                prev_done=0,
                prev_time=now,
                speed=0.0,
            )

    def job_update(self, file_id: int, status: str, size: int | bool | None = None, done: int | None = None) -> None:
        now = time.monotonic()
        with self._lock:
            if file_id not in self.active:
                return
            job = self.active[file_id]
            if done is not None:
                dt = now - job["prev_time"]
                if dt >= 0.5:
                    dd = done - job["prev_done"]
                    job["speed"] = dd / dt if dt > 0 else job["speed"]
                    job["prev_done"] = done
                    job["prev_time"] = now
                job["done"] = done
            job["status"] = status
            if size is not None:
                job["size"] = size

    def job_done(self, file_id: int, label: str, ok: bool, note: str = "") -> None:
        with self._lock:
            job = self.active.pop(file_id, None)
            if ok:
                self._total_done += 1
                if job and isinstance(job["size"], int):
                    self._total_bytes += job["size"]
            icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
            color = "green" if ok else "red"
            entry = f"{icon} [{color}]{label}[/{color}]"
            if note:
                entry += f"  [dim]{note}[/dim]"
            self.history.append(entry)

    def __rich__(self) -> Group:
        now = time.monotonic()

        with self._lock:
            snapshot = list(self.active.values())
            history_lines = list(self.history)
            elapsed_total = now - self._session_start
            done_count = self._total_done
            total_bytes = self._total_bytes

        # Active jobs table
        table = Table(box=box.SIMPLE, show_header=True, expand=True, header_style="bold dim", padding=(0, 1))
        table.add_column("", width=3)
        table.add_column("File")
        table.add_column("Size", width=10, justify="right")
        table.add_column("Speed", width=10, justify="right")
        table.add_column("Progress", width=26)

        for info in snapshot:
            st = info["status"]
            color = {"DL": "cyan", "UL": "yellow", "RT": "magenta"}.get(st, "white")
            size = info["size"]
            done = info["done"]
            speed = info["speed"]
            elapsed = now - info["start_time"]

            speed_str = f"[dim]{humanize.naturalsize(speed, gnu=True)}/s[/dim]" if speed > 0 else "[dim]—[/dim]"

            if size:
                pct = min(1.0, done / size)
                bar_w = 14
                filled = int(bar_w * pct)
                bar = (
                    f"[{color}]"
                    + "█" * filled
                    + f"[/{color}]"
                    + "[dim]"
                    + "░" * (bar_w - filled)
                    + "[/dim]"
                    + f" {pct * 100:4.0f}%"
                )
                size_str = humanize.naturalsize(size)
            else:
                spin = _SPINNER[int(now * 8) % len(_SPINNER)]
                elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                bar = f"[{color}]{spin}[/{color}] [dim]{humanize.naturalsize(done)} — {elapsed_str}[/dim]"
                size_str = "[dim]?[/dim]"

            table.add_row(
                f"[{color}]{st}[/{color}]",
                info["label"],
                size_str,
                speed_str,
                bar,
            )

        # Session stats footer
        h = int(elapsed_total // 3600)
        m = int((elapsed_total % 3600) // 60)
        s = int(elapsed_total % 60)
        session_str = f"{h:02d}:{m:02d}:{s:02d}"
        stats = Text.from_markup(
            f"[dim]Session: {session_str}  |  "
            f"Completed: {done_count}  |  "
            f"Transferred: {humanize.naturalsize(total_bytes)}[/dim]"
        )

        parts: list = []
        if history_lines:
            parts.extend(Text.from_markup(line) for line in history_lines)
            parts.append(Rule(style="dim"))
        parts.append(table)
        parts.append(Rule(style="dim"))
        parts.append(stats)
        return Group(*parts)
