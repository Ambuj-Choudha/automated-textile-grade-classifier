from app.ui.screens.mode_select import show_mode_selector
from app.ui.screens.login import handle_training_login
from app.ui.screens.operator import handle_operator_name
from app.ui.screens.capture_screen import show_grading_mode, show_training_mode


SCREEN_REGISTRY = {
    None:                  show_mode_selector,
    "training_requested":  handle_training_login,
    "grading_requested":   handle_operator_name,
    "training":            show_training_mode,
    "grading":             show_grading_mode,
}

__all__ = [
    "SCREEN_REGISTRY",
    "show_mode_selector",
    "handle_training_login",
    "handle_operator_name",
    "show_grading_mode",
    "show_training_mode",
]
