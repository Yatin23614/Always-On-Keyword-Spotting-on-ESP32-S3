from kws.labels import COMMAND_LABELS
from kws.policy import CommandGate

WAKE = [0.9, 0.1]
UNKNOWN = [0.0, 1.0]


def test_gate_emits_once_until_unknown_or_cooldown():
    gate = CommandGate(COMMAND_LABELS, threshold=.8, cooldown_ms=800)
    assert gate.update(WAKE, 0) == "hello_jarvis"
    assert gate.update(WAKE, 100) is None
    assert gate.update(WAKE, 801) == "hello_jarvis"
    assert gate.update(UNKNOWN, 900) is None
    assert gate.update(WAKE, 901) == "hello_jarvis"


def test_gate_rejects_low_confidence():
    assert CommandGate(COMMAND_LABELS).update([.79, .21], 0) is None
