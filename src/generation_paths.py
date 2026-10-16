"""Campaign location policy, with explicit registered execution support."""
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def resolve_campaign(value, project_root=None):
    project = Path(project_root or PROJECT_ROOT).resolve()
    explicit = os.environ.get('PSLG_EXECUTION_DIR')
    if explicit:
        target = Path(explicit).resolve()
        if not target.is_relative_to(project/'runs') or not (target/'run.json').is_file():
            raise ValueError('PSLG_EXECUTION_DIR must be a registered runs execution')
        return target
    requested = Path(value).expanduser()
    if not str(value).strip() or '..' in requested.parts:
        raise ValueError('Campaign must be a name or a path without parent traversal')
    candidate = requested if requested.is_absolute() else project/requested
    if candidate.exists():
        if not candidate.is_dir(): raise ValueError('Campaign is not a directory')
        return candidate.resolve()
    # Compatibility API for old clients; production launches use registered executions.
    if len(requested.parts)==1 and not requested.is_absolute():
        return project/'runs/generation'/requested
    target=candidate.resolve()
    if not target.is_relative_to(project/'runs'):
        raise ValueError('New campaigns must be under runs/')
    return target
