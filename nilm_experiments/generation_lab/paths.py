"""Support both project-root and legacy generation_lab module entry points."""
if __package__ == 'generation_lab':
    from run_paths import resolve_campaign
else:
    from ..run_paths import resolve_campaign
