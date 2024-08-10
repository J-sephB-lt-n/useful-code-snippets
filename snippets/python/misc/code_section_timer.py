"""
TAGS: code|code timer|script|script timer|section|sections|step|steps|time|timer
DESCRIPTION: Convenient interface for timing a script which contains multiple sections of interest
"""

import datetime
import time
from typing import Union


class CodeSectionTimer:
    """Convenient interface for timing a script which contains multiple sections of interest

    Example:
    >>> import time
    >>> timer = CodeSectionTimer()
    >>> timer.checkpoint("Started process")
    >>> time.sleep(1)
    >>> timer.checkpoint("Loading data")
    >>> time.sleep(3)
    >>> timer.checkpoint("Doing AI")
    >>> time.sleep(2)
    >>> timer.checkpoint("Process completed")
    >>> print( timer.summary_string() )
        [Started process] 2024-05-26 23:34:20
                                        1 seconds
        [Loading data] 2024-05-26 23:34:21
                                        3 seconds
        [Doing AI] 2024-05-26 23:34:24
                                        2 seconds
        [Process completed] 2024-05-26 23:34:26
    """

    def __init__(self) -> None:
        """Run at class instantiation"""
        self.history = []

    def checkpoint(self, name: str) -> None:
        """docstring TODO"""
        self.history.append((name, time.perf_counter(), datetime.datetime.now()))

    def n_seconds_to_human_readable(
        self, n_seconds: Union[int, float]
    ) -> tuple[int, int, int, int]:
        """Converts number of seconds into number of days, hours, minutes, seconds
        (remaining fraction of a second is discarded)

        Example:
            >>> n_seconds_to_human_readable(5_979_725)
            (69, 5, 2, 5)
        """
        n_seconds_in_a = {"day": 60 * 60 * 24, "hour": 60 * 60, "minute": 60}
        n_seconds_accounted_for: int = 0
        n_days: int = int(n_seconds / n_seconds_in_a["day"])
        n_seconds_accounted_for += n_days * n_seconds_in_a["day"]
        n_hours: int = int(
            (n_seconds - n_seconds_accounted_for) / n_seconds_in_a["hour"]
        )
        n_seconds_accounted_for += n_hours * n_seconds_in_a["hour"]
        n_minutes: int = int(
            (n_seconds - n_seconds_accounted_for) / n_seconds_in_a["minute"]
        )
        n_seconds_accounted_for += n_minutes * n_seconds_in_a["minute"]
        n_seconds_remain: int = int(n_seconds - n_seconds_accounted_for)

        return n_days, n_hours, n_minutes, n_seconds_remain

    def summary_string(self) -> str:
        """Builds summary of all timed sections (checkpoints) in history"""
        build_str = ""
        grand_total_seconds: int = 0
        for idx, chkpnt in enumerate(self.history):
            if idx > 0:
                secs_elapsed = chkpnt[1] - self.history[idx - 1][1]
                grand_total_seconds += secs_elapsed
                days, hours, minutes, seconds = self.n_seconds_to_human_readable(
                    secs_elapsed
                )
                build_str += f"\t\t"
                if days > 0:
                    build_str += f"{days} days "
                if hours > 0:
                    build_str += f"{hours} hours "
                if minutes > 0:
                    build_str += f"{minutes} minutes "
                build_str += f"{seconds} seconds\n"
            build_str += (
                f"[{chkpnt[0]}] " + chkpnt[2].strftime("%Y-%m-%d %H:%M:%S") + "\n"
            )
        total_days, total_hours, total_minutes, total_seconds = (
            self.n_seconds_to_human_readable(grand_total_seconds)
        )
        build_str += "\nTOTAL time: "
        if total_days > 0:
            build_str += f"{total_days} days "
        if total_hours > 0:
            build_str += f"{total_hours} hours "
        if total_minutes > 0:
            build_str += f"{total_minutes} minutes "
        build_str += f"{total_seconds} seconds\n"

        return build_str


if __name__ == "__main__":
    timer = CodeSectionTimer()
    timer.checkpoint("Started process")
    time.sleep(1)
    timer.checkpoint("Loading data")
    time.sleep(72)
    timer.checkpoint("Doing AI")
    time.sleep(222)
    timer.checkpoint("Process completed")
    print(timer.summary_string())
