from abc import ABC, abstractmethod
from collections import deque

class ContextDetector(ABC):
    """
    Class to be inherited by general swoks modules -
    KS, CvM, AD, CUSUM, FELT, etc.
    """
    def __init__(self, main_hist, wass_hist=None, ot_alg=None, needs_updating=False,\
                 adj=1):
        self.hist = main_hist
        self.wass_hist = wass_hist
        self.ot = ot_alg
        self.needs_updating = needs_updating
        self.adj = adj

    @abstractmethod
    def update(self, data=None):
        """
        Mainly for CPD methods that maintain internal history/storage/etc
        """
        return

    @abstractmethod
    def pval(self):
        print("You haven't made a pval returner.")
        return 1

    def get_wass(self):
        if self.ot is not None:
            return self.ot(self.hist.old_window(), self.hist.new_window())

    def set_task(self, task_label):
        self.hist.set_task(task_label)
        self.wass_hist.set_task(task_label)

class HistoryManager():
    """
    Maintains a simple history of max length.
    Deque automatically keeps length of history correct.
    Update outside of specific ContextDetector to prevent
    duplicated history entries.

    Old data is funnelled into hist_dict of current task.
    Dynamically updates to contain the desired number of tasks.

    h_len is like a "window" length.
    num_wins is how many windows you want available before the data is "archived".
    You can provide maxlen instead of num_wins if you want an exact history length.
    """
    def __init__(self, h_len=None, num_wins=None, maxlen=None):
        assert h_len is not None and (maxlen is not None or num_wins is not None),\
            "must provide h_len and either wass_len or maxlen"

        # Initialise history
        if num_wins is None:
            self.history = deque(maxlen=maxlen)
        else:
            self.history = deque(maxlen=h_len*num_wins)

        # For retrieving windows of old and new history.
        self.h_len = h_len

        # Initialise Task memories
        self.hist_dict = {0:deque(maxlen=self.h_len)}

    def __len__(self):
        return len(self.history)

    def update(self, data):
        self.hist_dict[task].update(self.history[0])
        self.history.update(data)

    def new_window(self):
        return self.history[-self.h_len:]

    def old_window(self):
        return self.hist_dict[self.task]

    def get_task_data(self, task):
        return self.hist_dict[task]

    def set_task(self, task):
        if task == self.task:
            # Don't update anything, it's the same task!!
            return

        # Trim history
        self.history = self.history[-self.h_len:]

        if task in self.hist_dict.keys:
            self.task = task
        else:
            # Make new task and new history.
            self.hist_dict[task] = deque(maxlen=self.h_len)

class cd_ks(ContextDetector):
    def __init__(self, main_hist, wass_hist):
        super().__init__(main_hist, wass_hist)

    def update(self):
        pass

    def pval(self):
        # TODO: change task functionality
        return stats.ks_2samp(self.adj*self.hist.old_window(),\
                              self.hist.new_window(),"greater").pvalue

class cd_ad(ContextDetector):
    # Blah

class cd_cvm(ContextDetector):
    # Blah
