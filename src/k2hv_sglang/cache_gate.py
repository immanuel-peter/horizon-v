"""Real scheduler/worker projector-update and fresh-engine image-cache verification."""
import argparse
import base64
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .smoke import post
from .native_gates import traces


def request(args,case):
    post(args.url,'/flush_cache',{})
    before=set(args.trace.glob('*.pt'))
    response=post(args.url,'/generate',{'input_ids':case['input_ids'],'image_data':case['image_data'],'sampling_params':{'max_new_tokens':2,'temperature':0},'return_logprob':True,'logprob_start_len':0})
    events=[torch.load(p,weights_only=False) for p in sorted(set(args.trace.glob('*.pt'))-before)]
    return response,[e for e in events if e['kind']=='vision']


def state(args):
    return [e for e in traces(args.trace).values() if e['kind']=='weights'][-1]


def load_projector():
    root=Path('hf/K2-Horizon-V-init');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];weights={}
    for key,shard in index.items():
        if key.startswith('projector.'):
            with safe_open(str(root/shard),framework='pt') as file:weights[key]=file.get_tensor(key).clone()
    return weights


def checkpoint(args,weights):
    source=Path('hf/K2-Horizon-V-init');target=args.directory/'updated-checkpoint';target.mkdir(exist_ok=True)
    index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    changed={index[k] for k in weights}
    for file in source.iterdir():
        if not file.is_file() or file.name.startswith('.') or file.suffix=='.pyc' or file.name=='SHA256SUMS':continue
        dest=target/file.name
        if dest.exists():continue
        if file.suffix=='.safetensors' and file.name not in changed:os.link(file,dest)
        elif file.suffix!='.safetensors':shutil.copyfile(file,dest)
    for shard in changed:
        with safe_open(str(source/shard),framework='pt') as file:values={k:weights.get(k,file.get_tensor(k)) for k in file.keys()}
        save_file(values,str(target/shard),metadata={'format':'pt'})
    save_file(weights,str(args.directory/'updated-projector.safetensors'))
    return target


def update(args):
    cases=json.loads((args.cases/'cases.json').read_text());case=next(c for c in cases if c['name']=='square')
    reset=post(args.url,'/update_weights_from_disk',{'model_path':str(Path('hf/K2-Horizon-V-init').resolve()),'load_format':'auto'})
    assert reset['success'],reset
    before=state(args)
    initial,first=request(args,case);repeat,second=request(args,case)
    assert len(first)==1 and not second, 'repeat image did not hit projected-feature cache'
    multiple=next(c for c in cases if c['name']=='multiple');request(args,multiple)
    from PIL import Image
    import io
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained('hf/K2-Horizon-V-init',trust_remote_code=True,local_files_only=True)
    import numpy as np
    patch_groups=np.random.default_rng(17).integers(0,256,(80,2,2,14,14,3),dtype=np.uint8)
    geometry=[]
    for size in ((560,112),(112,560)):
        width,height=size
        pixels=patch_groups.reshape(height//28,width//28,2,2,14,14,3).transpose(0,2,4,1,3,5,6).reshape(height,width,3)
        image=Image.fromarray(pixels)
        encoded=processor.image_processor([image],return_tensors='pt')
        stream=io.BytesIO();image.save(stream,format='PNG')
        control=dict(next(c for c in cases if c['name']=='wide'))
        control['image_data']=[base64.b64encode(stream.getvalue()).decode()]
        response,computed=request(args,control)
        assert len(computed)==1, 'different grid reused an image embedding'
        geometry.append((encoded,computed[0]))
    assert torch.equal(geometry[0][0]['pixel_values'],geometry[1][0]['pixel_values'])
    assert not torch.equal(geometry[0][0]['image_grid_thw'],geometry[1][0]['image_grid_thw'])
    assert not torch.equal(geometry[0][1]['projected'],geometry[1][1]['projected'])
    weights=load_projector();weights['projector.fc2.bias'][0]+=0.25
    target=checkpoint(args,weights)
    from sglang.srt.utils import MultiprocessingSerializer
    tensors=[(n,t.cuda()) for n,t in weights.items()]
    encoded=base64.b64encode(MultiprocessingSerializer.serialize(tensors)).decode()
    post(args.url,'/pause_generation',{'mode':'abort'})
    opened=post(args.url,'/begin_weight_update',{'sync_base':True})
    assert opened['success'],opened
    succeeded=False
    try:
        result=post(args.url,'/update_weights_from_tensor',{'serialized_named_tensors':[encoded],'flush_cache':True,'abort_all_requests':True,'weight_version':'step04-projector-only'})
        assert result['success'],result
        succeeded=True
    finally:
        closed=post(args.url,'/end_weight_update',{'abort':not succeeded})
        post(args.url,'/continue_generation',{})
        assert closed['success'],closed
    after=state(args)
    assert before['frozen_sha256']==after['frozen_sha256']
    assert after['names'] and all(n.startswith('projector.') for n in after['names']),after['names']
    updated,vision=request(args,case)
    assert len(vision)==1, 'projected features were not recomputed through the worker after update'
    delta=(vision[0]['projected'].float()-first[0]['projected'].float()).abs().max().item()
    assert delta>0.1,delta
    _,again=request(args,case);assert not again
    torch.save({'before':before,'after':after,'initial':initial,'updated':updated,'initial_feature':first[0]['projected'],'updated_feature':vision[0]['projected'],'feature_max_change':delta,'equal_pixels_different_grid_verified':True,'checkpoint':str(target)},args.directory/'running-update.pt')
    print(json.dumps({'worker_update':'passed','frozen_unchanged':True,'feature_max_change':delta,'fresh_checkpoint':str(target)}),flush=True)


def fresh(args):
    old=torch.load(args.directory/'running-update.pt',weights_only=False)
    case=next(c for c in json.loads((args.cases/'cases.json').read_text()) if c['name']=='square')
    fresh_state=state(args);result,vision=request(args,case)
    assert fresh_state['frozen_sha256']==old['after']['frozen_sha256']
    assert len(vision)==1
    assert torch.equal(vision[0]['projected'],old['updated_feature'])
    a=old['updated']['meta_info']['input_token_logprobs'];b=result['meta_info']['input_token_logprobs']
    errors=[abs(x[0]-y[0]) for x,y in zip(a,b) if x[0] is not None and y[0] is not None]
    assert max(errors)==0, max(errors)
    restored=post(args.url,'/update_weights_from_disk',{'model_path':str(Path('hf/K2-Horizon-V-init').resolve()),'load_format':'auto'})
    assert restored['success'],restored
    current=state(args)
    assert current['frozen_sha256']==old['before']['frozen_sha256']
    assert all(torch.equal(v,current['projector'][k]) for k,v in old['before']['projector'].items())
    _,restored_feature=request(args,case)
    assert torch.equal(restored_feature[0]['projected'],old['initial_feature'])
    from k2hv.gate_provenance import capture as provenance
    report={'provenance':provenance(),'passed':True,'update_path':'HTTP pause -> begin_weight_update -> update_weights_from_tensor -> end_weight_update -> continue; tokenizer manager/scheduler/worker, flush_cache=True','frozen_sha256':current['frozen_sha256'],'feature_max_change':old['feature_max_change'],'fresh_feature_max_error':0,'fresh_logprob_max_error':max(errors),'init_reloaded':True,'repeat_image_cache_verified':True,'multi_image_verified':True,'equal_pixels_different_grid_verified':old['equal_pixels_different_grid_verified']}
    (args.directory/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['update','fresh']);p.add_argument('--directory',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('--cases',type=Path,default=Path('outputs/step04/native-math'));p.add_argument('--url',default='http://127.0.0.1:30000');a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True);(update if a.stage=='update' else fresh)(a)


if __name__=='__main__':main()
