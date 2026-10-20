"""Inspect or explicitly test one INSPIRE gripper, without any arm motion API."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

from gripper_control import (GripperConnection, action_request, check_port,
                             decode_feedback, gripper_config, inspect_port)
from runtime_paths import LOCK_DIR, PROJECT_DIR
from teleop_config import load_config

ROOT = PROJECT_DIR
from readonly_client import ARM_IPS, check_known_controller_processes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('right', 'left'), required=True)
    parser.add_argument('--config', type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--open', action='store_true')
    actions.add_argument('--close', action='store_true')
    actions.add_argument('--position', type=int, help='Absolute opening in controller units, 1..1000')
    parser.add_argument('--execute', action='store_true', help='Send the requested gripper action')
    args = parser.parse_args()
    action = 'open' if args.open else 'close' if args.close else args.position
    if args.position is not None and not 1 <= args.position <= 1000:
        parser.error('--position must be 1..1000')
    if args.execute and action is None:
        parser.error('--execute requires --open, --close or --position')
    try:
        config = load_config(args.config or ROOT/('configs/'+args.arm+'_rotation_teleop.json'))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if config['arm'] != args.arm:
        parser.error('Config selects the wrong arm')
    settings = gripper_config(config.get('gripper'))
    connection = GripperConnection(ARM_IPS[args.arm], execute=args.execute)
    try:
        with (LOCK_DIR/'.single_joint_step.lock').open('a') as global_lock, \
                (LOCK_DIR/('.pico_'+args.arm+'.lock')).open('a') as side_lock:
            fcntl.flock(global_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(side_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check_known_controller_processes()
            port = inspect_port(connection)
            print(json.dumps({'arm': args.arm, 'port': port}, ensure_ascii=False), flush=True)
            if action is None:
                return 0 if port['feedback']['valid'] else 2
            request = action_request(action, settings)
            print(json.dumps({'execute': args.execute, 'requested_action': request,
                              'speed_and_force_units': 'dimensionless_controller_parameters'}, ensure_ascii=False), flush=True)
            if not args.execute:
                return 0
            check_port(port)
            ack = connection.ask(request)
            print(json.dumps({'controller_ack': ack}, ensure_ascii=False), flush=True)
            if ack.get('state') is not True:
                raise RuntimeError('Controller rejected gripper action')
            # Observations only; reaching a force threshold can stop before full closure.
            deadline = time.monotonic()+3.
            latest = None
            while time.monotonic() < deadline:
                latest = decode_feedback(connection.ask({'command': 'get_gripper_state'}))
                print(json.dumps({'feedback': latest}, ensure_ascii=False), flush=True)
                if type(action) is int:
                    actual = latest['opening_controller_units']
                    reached = (latest['valid'] and (
                               latest['raw']['mode'] in (1, 2, 3, 6) and abs(actual-action) <= 10 or
                               latest['raw']['mode']==6 and actual > action))
                else:
                    reached = latest['valid'] and latest['raw']['mode'] in ((1,) if action=='open' else (2, 6))
                if reached:
                    return 0
                time.sleep(1/settings['feedback_hz'])
            raise RuntimeError('No valid idle/force-stop feedback; action outcome unverified; not retried')
    except BlockingIOError:
        print('Stop the selected arm teleop/recovery session before using this tool.', file=sys.stderr)
        return 1
    except Exception as exc:
        print('GRIPPER ERROR: '+str(exc), file=sys.stderr)
        return 1
    finally:
        connection.close()


if __name__ == '__main__':
    raise SystemExit(main())
