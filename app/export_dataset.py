"""Manual local export using real, separately pinned official dataset writers."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

os.environ.setdefault('HF_HOME','/tmp/pico-export-cache/huggingface')
os.environ.setdefault('HF_DATASETS_CACHE','/tmp/pico-export-cache/datasets')

from export_core import make_plan, snapshot_source

LOCKS = {
    'v2.1':{'commit':'0cf864870cf29f4738d3ade893e6fd13fbd7cdb5',
            'module':'lerobot.common.datasets.lerobot_dataset'},
    'v3.0':{'commit':'0b067df57d21d3a02d6c511f1609172fa39ac29b',
            'module':'lerobot.datasets.lerobot_dataset'}}


def load_adapter(version):
    if sys.version_info<(3,10):
        raise RuntimeError('actual export requires isolated Python >=3.10; use --check-only on control Python')
    lock=LOCKS[version]
    distribution=importlib.metadata.distribution('lerobot')
    origin=json.loads(distribution.read_text('direct_url.json') or '{}')
    actual=origin.get('vcs_info',{}).get('commit_id')
    archive_hash=origin.get('archive_info',{}).get('hashes',{}).get('sha256')
    pins=json.loads((Path(__file__).parent/'export_source_pins.json').read_text())
    verified_archive=archive_hash==pins[version]['sha256']
    if actual!=lock['commit'] and not verified_archive:
        raise RuntimeError('LeRobot commit does not match pinned adapter: expected '+lock['commit']+', got '+str(actual))
    module=importlib.import_module(lock['module'])
    if module.CODEBASE_VERSION!=version:
        raise RuntimeError('official writer schema differs from selected adapter')
    return module.LeRobotDataset,{'commit':lock['commit'],'archive_sha256':archive_hash,'module':lock['module'],
        'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}


def summarize_plan(plan):
    return dict(plan,episodes=[{k:v for k,v in ep.items() if k!='samples'} for ep in plan['episodes']])


def export(plan,version,output,repo_id):
    import numpy as np
    from PIL import Image
    dataset_class,environment=load_adapter(version)
    output=output.resolve()
    if output.exists():
        raise FileExistsError('output exists; choose a new directory: '+str(output))
    accepted=[ep for ep in plan['episodes'] if ep['accepted']]
    if not accepted:
        raise ValueError('no eligible episodes; run --check-only for rejection reasons')
    for ep in accepted:
        if snapshot_source(Path(ep['path']))!=ep['raw_sha256']:
            raise RuntimeError('source changed since snapshot: '+ep['episode_id'])
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=output.with_name(output.name+'.partial-'+str(uuid.uuid4()))
    first=accepted[0]
    names=[side+'.'+joint for side in ('left','right') for joint in ('J1','J2','J3','J4','J5','J6','gripper')]
    features={key:{'dtype':'float32','shape':(14,),'names':names} for key in ('observation.state','action')}
    image_shapes={}
    for camera in ('front','left_wrist','right_wrist'):
        with Image.open(Path(first['path'])/first['samples'][0]['images'][camera]) as image:
            width,height=image.size
        image_shapes[camera]=(height,width,3)
        features['observation.images.'+camera]={'dtype':'video','shape':(height,width,3),'names':['height','width','channels']}
    kwargs={'repo_id':repo_id,'fps':plan['profile']['fps'],'root':staging,'robot_type':'dual_rm65',
            'features':features,'use_videos':True,'video_backend':'pyav'}
    if version=='v3.0':
        kwargs.update(vcodec='h264',batch_encoding_size=1)
    dataset=dataset_class.create(**kwargs)
    provenance=(staging/'provenance.jsonl').open('w')
    all_states,all_actions=[],[]
    manifest=dict(summarize_plan(plan),format=version,environment=environment,complete=False,
        action_order=names,units={'joints':'rad_absolute_unwrapped','gripper':'controller_opening/1000'},
        derived_hold_count=sum(ep['derived_hold_count'] for ep in accepted))
    try:
        for index,episode in enumerate(accepted):
            for sample in episode['samples']:
                frame={'observation.state':np.asarray(sample['state'],dtype=np.float32),
                       'action':np.asarray(sample['action'],dtype=np.float32),'task':episode['task']}
                for camera,relative in sample['images'].items():
                    with Image.open(Path(episode['path'])/relative) as image:
                        array=np.asarray(image.convert('RGB')).copy()
                    if array.shape!=image_shapes[camera]:
                        raise ValueError('image dimensions change between episodes: '+camera)
                    frame['observation.images.'+camera]=array
                dataset.add_frame(frame)
                all_states.append(sample['state']);all_actions.append(sample['action'])
                provenance.write(json.dumps({'episode_index':index,'raw_episode_uuid':episode['episode_id'],
                    'raw_monotonic_ns':sample['raw_monotonic_ns'],'frame_index':sample['grid_index'],
                    'sources':sample['sources']},ensure_ascii=False)+'\n')
            if version=='v3.0':
                dataset.save_episode(parallel_encoding=False)
            else:
                dataset.save_episode()
        dataset.stop_image_writer()
        if version=='v3.0':
            dataset.finalize()
        provenance.close()
        stats={}
        for key,values in (('observation.state',all_states),('action',all_actions)):
            a=np.asarray(values,dtype=np.float64)
            stats[key]={name:getattr(a,name)(axis=0).tolist() for name in ('min','max','mean','std')}
            stats[key].update(q01=np.quantile(a,.01,axis=0).tolist(),q99=np.quantile(a,.99,axis=0).tolist(),count=len(a))
        (staging/'vector_statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
        # Actual official reader, every sample and every image. No upload API.
        os.environ['HF_HUB_OFFLINE']='1'
        reader=dataset_class(repo_id=repo_id,root=staging,video_backend='pyav')
        expected=sum(len(ep['samples']) for ep in accepted)
        if len(reader)!=expected or reader.meta.total_episodes!=len(accepted):
            raise RuntimeError('official reader episode/frame count mismatch')
        ep_lengths=[len(ep['samples']) for ep in accepted]
        offsets=np.cumsum([0]+ep_lengths)
        for index in range(expected):
            sample=reader[index]
            for key in features:
                value=sample[key]
                if key in ('observation.state','action'):
                    if tuple(value.shape)!=(14,) or not np.isfinite(value.numpy()).all():
                        raise RuntimeError('invalid decoded vector: '+key)
                    reference=(all_states if key=='observation.state' else all_actions)[index]
                    if not np.allclose(value.numpy(),reference,rtol=1e-6,atol=1e-6):
                        raise RuntimeError('decoded vector differs from selected raw data')
                else:
                    camera=key.rsplit('.',1)[1]
                    h,w,_=image_shapes[camera]
                    if tuple(value.shape)!=(3,h,w):
                        raise RuntimeError('invalid decoded RGB image shape: '+key)
            ep_index=int(np.searchsorted(offsets,index,side='right')-1)
            if int(sample['episode_index'])!=ep_index or int(sample['frame_index'])!=index-offsets[ep_index]:
                raise RuntimeError('official reader episode/frame index mismatch')
        for ep in accepted:
            if snapshot_source(Path(ep['path']))!=ep['raw_sha256']:
                raise RuntimeError('raw source changed during export')
        manifest.update(complete=True,reader_verification={'frames':expected,'episodes':len(accepted),'decoded_all_images':True})
        (staging/'export_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        os.rename(staging,output)
    except BaseException:
        provenance.close()
        try:
            dataset.stop_image_writer()
            if version=='v3.0':dataset.finalize()
        except Exception:
            pass
        (staging/'export_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        raise
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--format',choices=tuple(LOCKS),required=True)
    parser.add_argument('--fps',type=int,default=30)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--repo-id',default='local/pico_sorting')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--include-preview',action='store_true',help='Explicit diagnostic export; never claim real sent actions')
    args=parser.parse_args()
    if args.output and args.output.exists():
        parser.error('output exists; refusing overwrite')
    profile=json.loads(args.profile.read_text()) if args.profile else {}
    profile.update(fps=args.fps,include_preview=args.include_preview)
    plan=make_plan(args.source,profile)
    if args.check_only:
        print(json.dumps(summarize_plan(plan),ensure_ascii=False,indent=2))
        return 0 if any(ep['accepted'] for ep in plan['episodes']) else 2
    if args.include_preview:
        parser.error('preview can be inspected with --check-only; official training export requires EXECUTE raw records')
    if not args.output:
        parser.error('--output is required for actual export')
    print(export(plan,args.format,args.output,args.repo_id))
    return 0


if __name__=='__main__':raise SystemExit(main())
