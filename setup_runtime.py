"""Check local CBC executables and repair PuLP's Intel-only wheel on Apple Silicon."""
import errno
import os
from pathlib import Path
import platform
import subprocess
import sys


def prepare_pulp():
    import pulp

    target = Path(pulp.PULP_CBC_CMD.pulp_cbc_path)
    try:
        subprocess.run([str(target), '-stop'], check=True, capture_output=True, timeout=15)
        return False
    except OSError as error:
        if sys.platform != 'darwin' or platform.machine() != 'arm64' or error.errno != errno.EBADARCH:
            raise
    # The bundled Intel executable cannot run. Use the native, pinned CBC package.
    import cbcbox

    source = Path(cbcbox.cbc_dist_dir()) / 'bin' / 'cbc'
    prefix = Path(sys.prefix).resolve()
    if sys.prefix == sys.base_prefix or not target.resolve().is_relative_to(prefix):
        raise RuntimeError('Create a virtual environment first; global PuLP will not be changed.')
    subprocess.run([str(source), '-stop'], check=True, capture_output=True, timeout=15)
    backup = target.with_name(target.name + '.intel-backup')
    if backup.exists():
        raise RuntimeError('An earlier CBC backup exists; check the environment before retrying.')
    target.rename(backup)
    try:
        target.symlink_to(os.path.relpath(source, target.parent))
        subprocess.run([str(target), '-stop'], check=True, capture_output=True, timeout=15)
    except Exception:
        target.unlink(missing_ok=True)
        backup.rename(target)
        raise
    return True


if __name__ == '__main__':
    repaired = prepare_pulp()
    print('PuLP CBC: native Apple Silicon executable linked.' if repaired else 'PuLP CBC: ready.')
