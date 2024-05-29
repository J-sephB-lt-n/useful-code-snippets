"""
TAGS: code|code timer|script|script timer|section|sections|step|steps|time|timer
DESCRIPTION: Convenient interface for timing a script which contains multiple sections of interest
"""

import datetime
import time


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
        """docstring TODO"""
        self.history = []

    def checkpoint(self, name: str) -> None:
        """docstring TODO"""
        self.history.append((name, time.perf_counter(), datetime.datetime.now()))

    def summary_string(self) -> str:
        """docstring TODO"""
        build_str = ""
        for idx, chkpnt in enumerate(self.history):
            if idx > 0:
                secs_elapsed = chkpnt[1] - self.history[idx - 1][1]
                tot_hours = int(secs_elapsed / 3600)
                tot_minutes = int(secs_elapsed / 60) - (60 * tot_hours)
                tot_seconds = (
                    int(secs_elapsed) - (60 * tot_minutes) - (3600 * tot_hours)
                )
                build_str += f"\t\t"
                if tot_hours > 0:
                    build_str += f"{tot_hours} hours "
                if tot_minutes > 0:
                    build_str += f"{tot_minutes} minutes "
                build_str += f"{tot_seconds} seconds\n"
            build_str += (
                f"[{chkpnt[0]}] " + chkpnt[2].strftime("%Y-%m-%d %H:%M:%S") + "\n"
            )
        return build_str


if __name__ == "__main__":
    timer = CodeSectionTimer()
    timer.checkpoint("Started process")
    time.sleep(1)
    timer.checkpoint("Loading data")
    time.sleep(3)
    timer.checkpoint("Doing AI")
    time.sleep(2)
    timer.checkpoint("Process completed")
    print(timer.summary_string())
