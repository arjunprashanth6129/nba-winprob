"""Score game states with the trained network, in NumPy (no PyTorch needed)."""
import numpy as np

from .config import WEIGHTS_PATH
from .features import FEATURES


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # overflow-free


class Predictor:
    """Loads exported weights (see `model.export_weights`) and scores game states."""

    def __init__(self, path=WEIGHTS_PATH):
        w = np.load(path)
        assert list(w["features"]) == FEATURES, "weights were trained on a different feature set"
        self.mean, self.std = w["mean"], w["std"]
        n = sum(k.startswith("w") for k in w.files)
        self.layers = [(w[f"w{i}"].T.copy(), w[f"b{i}"]) for i in range(n)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        h = (np.asarray(X, dtype=np.float32) - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = h * _sigmoid(h)  # SiLU
        return _sigmoid(h[:, 0])

    def score_states(self, states: list[dict]) -> list[float]:
        """P(home win) per state; a finished game is exactly 1 or 0."""
        probs = self.predict(np.asarray([s["x"] for s in states], dtype=np.float32))
        out = []
        for s, p in zip(states, probs):
            if s.get("final"):
                p = 1.0 if s["home_score"] > s["away_score"] else 0.0
            out.append(float(p))
        return out
