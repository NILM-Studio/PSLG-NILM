"""Shared output policy; existing runs retain their historical location."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_campaign(value, project_root=None):
    """Names go to runs/generation; explicit existing paths remain readable."""
    project = Path(project_root or PROJECT_ROOT).resolve()
    requested = Path(value).expanduser()
    if not str(value).strip() or '..' in requested.parts:
        raise ValueError('Campaign must be a name or a path without parent traversal')
    candidate = requested if requested.is_absolute() else project / requested
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError('Campaign is not a directory')
        return candidate.resolve()
    if len(requested.parts) == 1 and not requested.is_absolute():
        return project / 'runs' / 'generation' / requested
    target = candidate.resolve()
    if not target.is_relative_to(project / 'runs'):
        raise ValueError('New campaigns must be under runs/: use a campaign name or an absolute runs/ path')
    return target
