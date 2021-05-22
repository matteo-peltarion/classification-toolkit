class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a
    given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss
                            improved. Default: 7
            verbose (bool): If True, prints a message for each validation
                            loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify
                           as an improvement. Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """

        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.delta = delta

        self.trace_func = trace_func

    def should_stop(self, loss):

        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.delta:

            self.counter += 1

            if self.verbose:

                self.trace_func(
                    f'EarlyStopping counter: {self.counter}'
                    f'out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0
