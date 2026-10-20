"""Run a pinned export image with source read-only and destination writable."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--format',choices=('v2.1','v3.0'),required=True)
    parser.add_argument('--profile',type=Path)
    args,_=parser.parse_known_args()
    source=args.source.resolve();output=args.output.resolve()
    if not source.is_dir():parser.error('source does not exist')
    if source==output or source in output.parents:
        parser.error('output must be outside the raw source directory')
    if output.exists():parser.error('output exists; refusing overwrite')
    image='pico-export:'+('v21' if args.format=='v2.1' else 'v30')
    docker=['docker']
    if subprocess.run(docker+['info'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        docker=['sudo','-n','docker']
    if subprocess.run(docker+['image','inspect',image],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        parser.error('missing export image '+image+'; see pico_teleop_dev/README.md for image builds')
    output.parent.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parent.parent
    rewritten=[];previous=None
    for value in sys.argv[1:]:
        if previous in ('--source','--output','--profile'):
            rewritten.append(str(Path(value).resolve()))
            previous=None
        elif any(value.startswith(k+'=') for k in ('--source','--output','--profile')):
            key,path=value.split('=',1);rewritten.append(key+'='+str(Path(path).resolve()))
        else:
            rewritten.append(value);previous=value
    # Paths passed to the exporter are absolute. Mounting the caller's cwd is
    # unnecessary and made /tmp read-only when invoked from /tmp, breaking the
    # official reader's caches and temporary video encoding files.
    mounts={'/workspace':(project,'ro'),str(source):(source,'ro'),
            str(output.parent):(output.parent,'rw')}
    if args.profile:
        profile=args.profile.resolve();mounts[str(profile)]=(profile,'ro')
    command=docker+['run','--rm','--network','none','--user',str(os.getuid())+':'+str(os.getgid()),
                   '--workdir','/workspace','-e','HF_HUB_OFFLINE=1',
                   '-e','HF_HOME=/tmp/pico-export-cache/huggingface',
                   '-e','HF_DATASETS_CACHE=/tmp/pico-export-cache/datasets']
    for target,(host,mode) in mounts.items():
        command.extend(['-v',str(host)+':'+target+':'+mode])
    exporter=Path(__file__).with_name('export_dataset.py').resolve().relative_to(project)
    command.extend(['--entrypoint','/opt/export/bin/python',image,str(Path('/workspace')/exporter)]+rewritten)
    return subprocess.call(command)


if __name__=='__main__':raise SystemExit(main())
