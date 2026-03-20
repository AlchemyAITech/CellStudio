class Hook:
    """Base class for all runner hooks."""
    def before_run(self, runner, **kwargs):
        pass

    def after_run(self, runner, **kwargs):
        pass

    def before_train_epoch(self, runner, **kwargs):
        pass

    def before_train_iter(self, runner, **kwargs):
        pass

    def after_train_iter(self, runner, **kwargs):
        pass

    def after_train_epoch(self, runner, **kwargs):
        pass

    def before_val_epoch(self, runner, **kwargs):
        pass

    def before_val_iter(self, runner, **kwargs):
        pass

    def after_val_iter(self, runner, **kwargs):
        pass

    def after_val_epoch(self, runner, **kwargs):
        pass
