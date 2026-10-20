"""Bounded, read-only JSON queries for the two known RM65 controllers."""
import json
import math
import socket
import time
from pathlib import Path

ARM_IPS = {"left": "169.254.128.18", "right": "169.254.128.19"}
PORT = 8080
READ_QUERIES = {
    "get_arm_software_info": "arm_software_info",
    "get_arm_power_state": "arm_power_state",
    "get_current_arm_state": "current_arm_state",
}


class ReadOnlyError(RuntimeError):
    pass


def check_known_controller_processes():
    """A conservative local check; it cannot detect every other control source."""
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for item in proc.iterdir():
        if not item.name.isdigit():
            continue
        try:
            name = (item / "comm").read_text().strip()
        except (OSError, UnicodeError):
            continue
        if name in {"atom", "rm_driver", "arm_driver"}:
            raise ReadOnlyError("Detected control process %s (PID %s); finish that control session first." % (name, item.name))


def query_one(host, command, port=PORT, timeout=2.0):
    # Validate BEFORE opening any connection. No arbitrary payload is accepted.
    if command not in READ_QUERIES:
        raise ReadOnlyError("Command is not in the read-only allowlist: %s" % command)
    if not math.isfinite(timeout) or not 0.2 <= timeout <= 5.0:
        raise ReadOnlyError("Timeout must be between 0.2 and 5 seconds.")
    payload = (json.dumps({"command": command}, separators=(",", ":")) + "\r\n").encode("ascii")
    expected = READ_QUERIES[command]
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.sendall(payload)
        deadline = time.monotonic() + timeout
        buffer = b""
        total_bytes = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ReadOnlyError("Timed out waiting for %s from %s." % (command, host))
            connection.settimeout(remaining)
            chunk = connection.recv(4096)
            if not chunk:
                raise ReadOnlyError("Connection closed before a complete matching reply.")
            total_bytes += len(chunk)
            if total_bytes > 65536:
                raise ReadOnlyError("Reply exceeded the 64 KiB limit.")
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    response = json.loads(line)
                except (ValueError, UnicodeError) as exc:
                    raise ReadOnlyError("Malformed JSON reply: %s" % exc)
                if not isinstance(response, dict):
                    raise ReadOnlyError("Expected a JSON object.")
                if response.get("state") == expected or response.get("command") in {expected, command}:
                    return response


def summarize_arm_state(response):
    state = response.get("arm_state")
    if not isinstance(state, dict):
        raise ReadOnlyError("Missing arm_state object.")
    joints = state.get("joint")
    if not isinstance(joints, list) or len(joints) != 6:
        raise ReadOnlyError("Expected six joint angles for this RM65 platform.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in joints):
        raise ReadOnlyError("Invalid joint feedback.")
    # JSON protocol units: joint values are 0.001 degree; SDK values are already degrees.
    result = {"joint_degrees": [value / 1000.0 for value in joints]}
    for key in ("arm_err", "sys_err", "err"):
        if key in state:
            result[key] = state[key]
    pose = state.get("pose")
    if isinstance(pose, list) and len(pose) == 6:
        if all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in pose):
            result["position_m"] = [value / 1000000.0 for value in pose[:3]]
            result["orientation_rad"] = [value / 1000.0 for value in pose[3:]]
    return result


def preview_joint_delta(joints, joint_number, delta_degrees):
    """Local arithmetic only. This function has no connection or send operation."""
    if len(joints) != 6 or not 1 <= joint_number <= 6:
        raise ValueError("Use a six-joint array and a joint number from 1 to 6.")
    values = list(joints)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Joint values must be finite numbers.")
    if not math.isfinite(delta_degrees) or abs(delta_degrees) > 0.5:
        raise ValueError("Offline preview delta must be finite and at most 0.5 degree.")
    values[joint_number - 1] += delta_degrees
    return values
