"""兼容原右臂恢复命令；共享实现见 recover_arm_initial.py。"""
import sys

from recover_arm_initial import load_initial, main, validate_reviewed


if __name__ == '__main__':
    sys.exit(main())
