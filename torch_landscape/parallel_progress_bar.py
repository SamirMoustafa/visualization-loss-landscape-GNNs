from joblib import Parallel
from tqdm.asyncio import tqdm


class ProgressParallel(Parallel):
    """
    A parallel implementation of joblib.Parallel which uses tqdm to display a progress bar.
    """

    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        """
        Initializes the ProgressParallel.

        :param use_tqdm: Boolean flag to enable/disable the progress bar.
        :param total: The total number of tasks to execute.
        :param args: Arguments for the Parallel class.
        :param kwargs: Keyword arguments for the Parallel class.
        """
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
