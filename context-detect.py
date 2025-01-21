from abc import ABC, abstractmethod
from collections import deque

class ContextDetector(ABC):
    def __init__(self, main_hist, wass_hist, ot_alg=None):
        if ot_alg is None:
            pass

class HistoryManager():
    """
    Maintains a simple history of max length.
    Deque automatically keeps length of history correct.
    Update outside of specific ContextDetector to prevent
    duplicated history entries.
    """
    def __init__(self, maxlen):
        self.history = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.history)

    def update(self, data):
        self.history.update(data)


class cd_ks(ContextDetector):
    def __init__(self, main_hist, wass_hist):
        super().__init__(main_hist, wass_hist)
    # Blah

class cd_ad(ContextDetector):
    # Blah

class cd_cvm(ContextDetector):
    # Blah
