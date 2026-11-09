"""Post-inference policy preventing repeated or low-confidence activations."""

from dataclasses import dataclass


@dataclass
class CommandGate:
    labels: tuple[str, ...]
    threshold: float = 0.80
    cooldown_ms: int = 800
    last_command: str | None = None
    last_emit_ms: int = -10**9

    def update(self, probabilities, now_ms: int) -> str | None:
        index = max(range(len(probabilities)), key=lambda i: probabilities[i])
        label = self.labels[index]
        confidence = float(probabilities[index])
        if label == "unknown" or confidence < self.threshold:
            self.last_command = None
            return None
        if label == self.last_command and now_ms - self.last_emit_ms < self.cooldown_ms:
            return None
        self.last_command = label
        self.last_emit_ms = now_ms
        return label
