# stop_detector.py
# Part of EmoNAVI-Kit: emotional learning components

class StopDetector:
    """
    Emotion-aware learning stagnation detector.
    Determines whether to suppress parameter injection based on recent loss fluctuations.
    """
    def __init__(self, tolerance: float = 1e-4, patience: int = 5):
        self.tolerance = tolerance      # Threshold for detecting stagnation
        self.patience = patience        # How many consecutive steps to observe
        self.counter = 0                # Number of steps within tolerance
        self.last_value = None

    def update(self, current_value: float) -> bool:
        """
        Update the detector with the current loss or scalar value.

        Args:
            current_value (float): The monitored value (e.g., emo_scalar or loss delta)

        Returns:
            bool: True if learning appears stagnant and updates should pause, else False
        """
        if self.last_value is None:
            self.last_value = current_value
            return False

        if abs(current_value - self.last_value) < self.tolerance:
            self.counter += 1
        else:
            self.counter = 0  # reset if movement resumes

        self.last_value = current_value
        return self.counter >= self.patience

    def reset(self):
        """Reset internal state (e.g., at epoch boundary)"""
        self.counter = 0
        self.last_value = None