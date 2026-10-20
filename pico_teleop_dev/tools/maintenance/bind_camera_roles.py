"""Explicit operator-provided serial binding; never infer a role from ordering."""
import argparse
import json
from pathlib import Path
import shutil
import time
from raw_episode_writer import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/cameras.json')
    for role in ('front','left-wrist','right-wrist'):
        parser.add_argument('--'+role,required=True)
    args=parser.parse_args()
    roles={'front':args.front,'left_wrist':args.left_wrist,'right_wrist':args.right_wrist}
    config=json.loads(args.config.read_text())
    if len(set(roles.values()))!=3 or set(roles.values())!=set(config['known_serials']):
        parser.error('provide three distinct known serials after physical identification')
    shutil.copy2(args.config,args.config.with_name(args.config.name+'.before-bind-'+str(time.time_ns())))
    for role,serial in roles.items():config['cameras'][role]['serial']=serial
    config['role_confirmation']='explicit_operator_binding'
    config['role_confirmation_wall_time_ns']=time.time_ns()
    atomic_json(args.config,config)
    print(json.dumps(config,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
