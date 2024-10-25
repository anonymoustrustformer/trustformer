class NoamLR:
    """ Learning rate schedule described in:
        'Attention is All You Need' https://arxiv.org/abs/1706.03762
    """
    def __init__(self, model_size, warmup_steps):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        """ Update current learning rate """
        self.step_num += 1
        lr = self.rate()
        return lr

    def rate(self, step=None):
        """ Calculate learning rate using Noam method """
        if step is None:
            step = self.step_num
        return self.model_size ** (-0.5) * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
