"""One horizontal reference shared by the two processes for a teleop session.

Resolution happens only when anchoring a Grip. Active control ticks do no file
I/O here. The supervisor gives each run a new private path; no old calibration
is reused after restarting teleop.
"""
import fcntl
import json
from pathlib import Path

from incremental import IncrementalMapper, MappingError


class SharedOperatorReference:
    def __init__(self, path):
        self.path = Path(path)
        self.basis = None

    def resolve(self, candidate):
        if self.basis is not None:
            return [row.copy() for row in self.basis]
        candidate = IncrementalMapper.checked_basis(candidate)
        with self.path.open('a+') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            encoded = stream.read()
            if encoded:
                document = json.loads(encoded)
                if document.get('schema') != 1:
                    raise MappingError('Invalid shared operator reference')
                basis = IncrementalMapper.checked_basis(document.get('operator_yaw_basis'))
            else:
                basis = candidate
                stream.write(json.dumps({'schema': 1, 'operator_yaw_basis': basis})+'\n')
                stream.flush()
        self.basis = basis
        return [row.copy() for row in basis]
