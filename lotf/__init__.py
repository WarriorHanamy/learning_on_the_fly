from pathlib import Path

_LOTF_DIR = Path(__file__).resolve().parent
LOTF_PATH = str(_LOTF_DIR)
LOTF_ROOT = _LOTF_DIR.parent


def resolve_path(path):
    """Resolve a path relative to project root if not absolute.

    Args:
        path: Path string or Path object.

    Returns:
        Absolute Path resolved against LOTF_ROOT.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (LOTF_ROOT / p).resolve()
