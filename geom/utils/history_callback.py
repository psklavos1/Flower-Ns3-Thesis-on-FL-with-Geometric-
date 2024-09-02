import keras


class CustomHistory(keras.callbacks.History):
    def __init__(self) -> None:
        super().__init__()
        self.batch_metrics = []

    def on_batch_end(self, batch, logs=None):
        self.batch_metrics.append(logs.copy())
        super().on_batch_end(batch, logs)
