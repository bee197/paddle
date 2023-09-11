
class ReplayMemory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_rpm(self):
        # act
        del self.actions[:]
        # obs
        del self.states[:]
        # log_b
        del self.logprobs[:]
        # reward
        del self.rewards[:]
        # done
        del self.is_terminals[:]

    def __call__(self, *args, **kwargs):
        print("is_done", self.is_terminals)
