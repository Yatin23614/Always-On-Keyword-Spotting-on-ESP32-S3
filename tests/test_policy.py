from kws.labels import COMMAND_LABELS
from kws.policy import CommandGate

def test_gate_emits_once_until_unknown_or_cooldown():
    gate = CommandGate(COMMAND_LABELS, threshold=.8, cooldown_ms=800)
    probabilities = [.9] + [0.0] * 8
    assert gate.update(probabilities, 0) == "start"
    assert gate.update(probabilities, 100) is None
    assert gate.update(probabilities, 801) == "start"
    assert gate.update([0.0] * 8 + [1.0], 900) is None
    assert gate.update(probabilities, 901) == "start"

def test_gate_rejects_low_confidence():
    assert CommandGate(COMMAND_LABELS).update([.79] + [0.0] * 8, 0) is None
