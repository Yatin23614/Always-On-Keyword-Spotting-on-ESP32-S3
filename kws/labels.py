"""The model label order is part of the firmware ABI; keep it stable."""

COMMAND_LABELS = (
    "start",
    "stop",
    "next",
    "previous",
    "volume_up",
    "volume_down",
    "lights_on",
    "lights_off",
    "unknown",
)

LABEL_TO_ID = {label: index for index, label in enumerate(COMMAND_LABELS)}
