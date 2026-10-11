"""Shared validation for file-replay options (not live camera settings)."""
import operator


def positive_ratio(value):
    """Accept a positive integer from a CLI, JSON config, or Python caller."""
    try:
        if isinstance(value, bool):
            raise ValueError
        ratio = int(value) if isinstance(value, str) else operator.index(value)
        if ratio <= 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError('ratio must be a positive integer') from None
    return ratio
