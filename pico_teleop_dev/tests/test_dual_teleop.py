"""Process isolation and stop propagation; no child or device is started."""
from pathlib import Path
import fcntl
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import dual_teleop


class DualSupervisorTests(unittest.TestCase):
    def run_fake(self,first_code=0,execute=False):
        actions=[];workers=[];commands=[]
        class Worker:
            def __init__(self,index):self.index=index;self.code=None
            def poll(self):return self.code
            def terminate(self):actions.append(('terminate',self.index));self.code=0
            def wait(self,**_):actions.append(('wait',self.index));return self.code
        def popen(command,**_):
            commands.append(command);p=Worker(len(workers));workers.append(p);return p
        def advance(_):workers[0].code=first_code
        result=dual_teleop.supervise({'left':Path('/tmp/left.json'),'right':Path('/tmp/right.json')},
            execute=execute,duration=5,popen=popen,sleep=advance)
        return result,commands,actions

    def test_preview_routes_two_configs_and_never_enables_execute(self):
        result,commands,actions=self.run_fake()
        self.assertEqual(result,0)
        self.assertIn('/tmp/left.json',commands[0]);self.assertIn('/tmp/right.json',commands[1])
        self.assertTrue(all('--preview'in c and '--execute'not in c for c in commands))
        references=[c[c.index('--operator-reference')+1] for c in commands]
        self.assertEqual(references[0],references[1])
        self.assertEqual(actions[0],('terminate',1))

    def test_explicit_execute_propagates_to_both_and_failure_stops_peer(self):
        result,commands,actions=self.run_fake(1,True)
        self.assertEqual(result,1)
        self.assertTrue(all('--execute'in c for c in commands))
        self.assertIn(('terminate',1),actions)

    def test_second_start_failure_terminates_first_worker_before_returning(self):
        actions=[]
        class Worker:
            code=None
            def poll(self):return self.code
            def terminate(self):actions.append('terminate');self.code=0
            def wait(self,**_):actions.append('wait');return self.code
        def popen(*_,**__):
            if actions:raise OSError('start failed')
            actions.append('start');return Worker()
        with self.assertRaises(OSError):
            dual_teleop.supervise({'left':Path('/tmp/l'),'right':Path('/tmp/r')},popen=popen)
        self.assertEqual(actions,['start','terminate','wait'])


class ArmLockTests(unittest.TestCase):
    def test_global_shared_workers_exclude_exclusive_recovery_and_same_arm(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            with (root/'global').open('a')as left,(root/'global').open('a')as right,(root/'global').open('a')as recovery:
                fcntl.flock(left,fcntl.LOCK_SH|fcntl.LOCK_NB)
                fcntl.flock(right,fcntl.LOCK_SH|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):fcntl.flock(recovery,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with (root/'left').open('a')as a,(root/'left').open('a')as duplicate,(root/'right').open('a')as b:
                    fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):fcntl.flock(duplicate,fcntl.LOCK_EX|fcntl.LOCK_NB)


if __name__=='__main__':unittest.main()
