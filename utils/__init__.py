# Expose optional AdvancedAI class without breaking package imports
# Try the current module name first, then silently disable if unavailable
try:
	from .advanced_ai import AdvancedAI  # preferred location
except Exception:
	AdvancedAI = None  # type: ignore
