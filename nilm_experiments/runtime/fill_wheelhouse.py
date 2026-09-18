"""Build/download only missing exact-version wheels; preserve existing CUDA wheels."""
import subprocess
import sys
from pathlib import Path
from packaging.utils import canonicalize_name, parse_wheel_filename

root = Path(__file__).resolve().parent
present = set()
for wheel in (root / 'wheelhouse').glob('*.whl'):
    name, version, _, _ = parse_wheel_filename(wheel.name)
    present.add((canonicalize_name(name), str(version)))
missing = []
for line in (root / 'requirements-lock.txt').read_text().splitlines():
    name, version = line.split('==', 1)
    if (canonicalize_name(name), version) not in present:
        missing.append(line)
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--timeout', '120', '--retries', '8', '--no-deps', '--wheel-dir', str(root / 'wheelhouse'), *missing], check=True)
