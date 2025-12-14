class HeapEntry:
    """
    Class used for heapq to store sequences of states/actions.
    """
    def __init__(self, priority, io_seq):
        """
        :param priority: Total reward for state-action sequence.
        :param io_seq: Sequence of states and actions (input and output arrays).
        """
        self.io_seq = io_seq
        self.priority = priority
    
    def __lt__(self, other):
        return self.priority <= other.priority