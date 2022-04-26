from typing import Callable


class Updater:
    """A neat little class to perform tasks periodically without maintaining separate counters"""
    def __init__(self, do_update: Callable[[], None]):
        self.do_update = do_update
        self.steps_since_update = 0

    def update_now(self):
        self.do_update()
        self.steps_since_update = 0

    def update_every(self, steps: int) -> bool:
        """
        Note that update_now resets the counter since the last update
        :return: Whether an update was performed
        """
        if self.steps_since_update >= steps - 1:
            self.update_now()
            return True
        else:
            self.steps_since_update += 1
            return False
