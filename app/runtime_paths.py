"""Shared deployment paths; old and new entrypoints use identical device locks."""
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
CONFIG_DIR = PROJECT_DIR / 'configs'
ROBOT_ROOT = Path(os.environ.get('PICO_ROBOT_ROOT', str(PROJECT_DIR.parent / 'RM'))).resolve()
LOCK_DIR = ROBOT_ROOT
RECEIVER_STATE_DIR = Path(os.environ.get('PICO_RECEIVER_STATE_DIR',
                                       str(ROBOT_ROOT / 'pico_xr/state'))).resolve()
RECEIVER_SOCKET = Path(os.environ.get('PICO_RECEIVER_SOCKET', str(RECEIVER_STATE_DIR / 'pico.sock')))
