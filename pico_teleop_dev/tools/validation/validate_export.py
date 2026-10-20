"""Read every image/vector with the pinned official reader and load a torch batch."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from export_dataset import load_adapter


def validate(path):
    manifest=json.loads((path/'export_manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('export has no complete commit')
    cls,environment=load_adapter(manifest['format'])
    reader=cls('local/pico_synthetic_smoke' if manifest.get('synthetic_smoke') else 'local/pico_sorting',
               root=path,video_backend='pyav')
    episodes=[ep for ep in manifest['episodes'] if ep['accepted']]
    provenance=[json.loads(s) for s in (path/'provenance.jsonl').read_text().splitlines()]
    max_rgb_error=0.
    for i,trace in enumerate(provenance):
        row=reader[i]
        episode=episodes[trace['episode_index']]
        raw=Path(episode['path'])
        if row.get('task')!=episode['task']:
            raise ValueError('task does not match source episode')
        for camera in ('front','left_wrist','right_wrist'):
            # Locate by original source sequence; no row-index alignment assumption.
            records=[json.loads(s) for s in (raw/'frames'/(camera+'.jsonl')).read_text().splitlines()]
            selected=next(r for r in records if r['source_sequence']==trace['sources'][camera]['source_sequence'])
            with Image.open(raw/selected['image_path']) as image:
                expected=np.asarray(image.convert('RGB')).mean(axis=(0,1))/255
            actual=row['observation.images.'+camera].numpy().mean(axis=(1,2))
            error=float(np.max(np.abs(expected-actual)))
            max_rgb_error=max(max_rgb_error,error)
            if error>.08:
                raise ValueError('RGB/view readback differs from source: '+camera)
    batch=next(iter(DataLoader(reader,batch_size=min(2,len(reader)),num_workers=0,shuffle=False)))
    return {'format':manifest['format'],'commit':environment['commit'],'frames':len(reader),
        'episodes':reader.meta.total_episodes,'all_tasks_and_rgb_views_verified':True,
        'maximum_rgb_channel_mean_error':max_rgb_error,
        'torch_batch_state_shape':list(batch['observation.state'].shape),
        'torch_batch_action_shape':list(batch['action'].shape),
        'torch_batch_image_shapes':{c:list(batch['observation.images.'+c].shape)
            for c in ('front','left_wrist','right_wrist')},
        'synthetic':manifest.get('synthetic_smoke',False),'model_training_run':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(validate(args.dataset),indent=2))
