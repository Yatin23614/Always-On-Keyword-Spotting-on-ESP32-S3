"""The model label order is part of the firmware ABI; keep it stable."""

COMMAND_LABELS = (
    "hello_jarvis",
    "unknown",
)

LABEL_TO_ID = {label: index for index, label in enumerate(COMMAND_LABELS)}
