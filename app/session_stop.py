"""One shared exit byte; initialization only touches files, gate reads use mmap."""
import mmap
import os


class SessionStopFlag:
    def __init__(self, path, create=False):
        self.fd = os.open(str(path), os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0), 0o600)
        if create:
            os.ftruncate(self.fd, 1)
        self.memory = mmap.mmap(self.fd, 1)

    def is_set(self):
        return bool(self.memory[0])

    def set(self):
        self.memory[0] = 1

    def close(self):
        self.memory.close()
        os.close(self.fd)
