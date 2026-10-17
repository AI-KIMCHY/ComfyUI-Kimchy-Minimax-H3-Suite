from .kimchy_upscaler_v2 import KimchyH3ZeroSplitV2
from .kimchy_upscaler_mp import KimchyH3ZeroSplitMP
from .kimchy_scheduler import KimchyH3DualScheduler

NODE_CLASS_MAPPINGS = {
    "KimchyH3ZeroSplitV2": KimchyH3ZeroSplitV2,
    "KimchyH3ZeroSplitMP": KimchyH3ZeroSplitMP,
    "KimchyH3DualScheduler": KimchyH3DualScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KimchyH3ZeroSplitV2": "🌶️ Kimchy H3 Upscaler V2",
    "KimchyH3ZeroSplitMP": "🌶️ Kimchy H3 Upscaler (Side-MP 1MP~4K)",
    "KimchyH3DualScheduler": "🌶️ Kimchy H3 Dual Scheduler",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
