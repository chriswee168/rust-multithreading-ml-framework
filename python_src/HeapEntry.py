class HeapEntry:
    """
    Class used for heapq to store sequences of states/actions.
    """
    def __init__(self, priority, entry):
        """
        :param priority: Total reward for state-action sequence.
        :param entry: A single state and action pair (input and output array).
        """
        self.entry = entry
        self.priority = priority
    
    def __lt__(self, other):
        return self.priority <= other.priority