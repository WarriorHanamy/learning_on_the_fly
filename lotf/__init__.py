from pathlib import Path

LOTF_PATH = Path(__file__).resolve().parent
LOTF_ROOT = LOTF_PATH.parent


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
