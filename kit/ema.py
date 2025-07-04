# ema.py
# Part of EmoNAVI-Kit: emotional learning components

class EMA:
    """
    Exponential Moving Average (EMA) tracker.
    Suitable for tracking scalar values like loss over different time scales.
    """
    def __init__(self, decay: float, init_value: float = 0.0):
        self.decay = decay
        self.value = init_value
        self.initialized = False

    def update(self, x: float) -> float:
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.decay * self.value + (1 - self.decay) * x
        return self.value

    def get(self) -> float:
        return self.value