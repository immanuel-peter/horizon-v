"""Full-path HF/native capture, preprocessing, vision and logprob gates on CUDA."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .smoke import post
from .logprob_gate import MAX_ERROR, MEAN_ERROR


def fixtures():
    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for w, h in ((448,448), (560,112), (112,560), (1,1), (1800,1800), (112,112))]
    for name, pictures in zip(("square", "wide", "tall", "minimum", "maximum", "multiple"), [[x] for x in images[:5]] + [[images[1], images[5]]]):
        yield name, pictures


def capture(args):
    from k2hv.gpu_numerics import enable
    enable()
    p = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    m = AutoModelForImageTextToText.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    cases = []
    for name, images in fixtures():
        content = [{"type":"image", "image":im} for im in images] + [{"type":"text", "text":"Describe these images."}]
        messages = [{"role":"user", "content":content}, {"role":"assistant", "think":"", "content":"Colored shapes."}]
        inputs = p.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
        towers, projected = [], []
        lengths = inputs['image_grid_thw'].prod(-1).tolist()
        with torch.no_grad():
            for pix, grid in zip(inputs['pixel_values'].split(lengths), inputs['image_grid_thw']):
                tower = m.visual(pix.cuda().bfloat16(), grid[None].cuda())
                towers.append(tower.cpu()); projected.append(m.projector(tower).cpu())
            ids = inputs['input_ids'].cuda()
            embed = m.get_input_embeddings()(ids)
            embed[ids == p.image_token_id] = torch.cat(projected).cuda()
            logits = m(**{k:v.cuda() for k,v in inputs.items()}, use_cache=False).logits[0,:-1].float()
            lp = logits.log_softmax(-1).gather(-1,ids[0,1:,None]).flatten().cpu()
        media=[]
        for im in images:
            stream=io.BytesIO(); im.save(stream, format='PNG'); media.append(base64.b64encode(stream.getvalue()).decode())
        path=args.directory/f'{name}.pt'
        torch.save({**dict(inputs), 'tower':towers, 'projected':projected, 'embeddings':embed.cpu(), 'logprobs':lp},path)
        cases.append({'name':name,'input_ids':inputs['input_ids'][0].tolist(),'image_data':media,'tensor_file':path.name})
    args.directory.joinpath('cases.json').write_text(json.dumps(cases))
    from k2hv.gate_provenance import capture as provenance
    args.directory.joinpath('hf-provenance.json').write_text(json.dumps(provenance(),indent=2)+'\n')


def measure(a,b,atol,rtol=0):
    if a.shape!=b.shape:
        return {'passed':False, 'shape_a':list(a.shape), 'shape_b':list(b.shape)}
    d=(a.float()-b.float()).abs()
    return {'passed':bool(torch.allclose(a.float(),b.float(),atol=atol,rtol=rtol)), 'max_abs':d.max().item(), 'mean_abs':d.mean().item(), 'atol':atol,'rtol':rtol}


def traces(directory):
    return {p:torch.load(p,weights_only=False,map_location='cpu') for p in directory.glob('*.pt')}


def evaluate(args):
    cases=json.loads((args.directory/'cases.json').read_text()); results=[]
    for case in cases:
        expected=torch.load(args.directory/case['tensor_file'],weights_only=False,map_location='cpu')
        for mode in ('expanded','raw'):
            post(args.url,'/flush_cache',{})
            before=set(args.trace.glob('*.pt'))
            ids=case['input_ids']
            if mode=='raw':
                ids=[token for i,token in enumerate(ids) if token!=250062 or not i or ids[i-1]!=250062]
            result=post(args.url,'/generate',{'input_ids':ids,'image_data':case['image_data'],'sampling_params':{'max_new_tokens':1,'temperature':0},'return_logprob':True,'logprob_start_len':0})
            events=[torch.load(p,weights_only=False,map_location='cpu') for p in sorted(set(args.trace.glob('*.pt'))-before)]
            processors=[e for e in events if e['kind']=='processor']
            vision=[e for e in events if e['kind']=='vision']
            row={'case':case['name'],'mode':mode,'checks':{}}
            checks=row['checks']; proc=processors[-1]
            checks['ids']={'passed':list(proc['ids'])==case['input_ids']}
            checks['pixels']=measure(torch.cat(proc['pixels']),expected['pixel_values'],0)
            checks['grids']=measure(proc['grids'],expected['image_grid_thw'],0)
            checks['feature_count']={'passed':case['input_ids'].count(250062)==expected['pixel_values'].shape[0]//4==sum(v['projected'].shape[0] for v in vision)}
            misses = len(vision)
            candidates = [e for e in traces(args.trace).values() if e['kind']=='vision']
            vision = [next(e for e in reversed(candidates) if torch.equal(e['grid'], grid[None]) and torch.equal(e['pixels'], pix)) for pix,grid in zip(expected['pixel_values'].split(expected['image_grid_thw'].prod(-1).tolist()),expected['image_grid_thw'])]
            checks['image_count']={'passed':len(vision)==len(expected['tower']), 'computed_this_request':misses}
            checks['feature_count']={'passed':case['input_ids'].count(250062)==expected['pixel_values'].shape[0]//4==sum(v['projected'].shape[0] for v in vision)}
            checks['tower']=[measure(v['tower'],x,0.05,0.03) for v,x in zip(vision,expected['tower'])]
            checks['projected']=[measure(v['projected'],x,0.05,0.03) for v,x in zip(vision,expected['projected'])]
            rows=result['meta_info']['input_token_logprobs']
            assert len(rows)==len(case['input_ids'])
            assert all(r[1]==case['input_ids'][i] and r[0] is not None for i,r in enumerate(rows) if i and case['input_ids'][i]!=250062)
            errors=[abs(r[0]-expected['logprobs'][i-1].item()) for i,r in enumerate(rows) if i and case['input_ids'][i]!=250062]
            checks['logprobs']={'passed':max(errors)<=MAX_ERROR and sum(errors)/len(errors)<=MEAN_ERROR,'max_abs':max(errors),'mean_abs':sum(errors)/len(errors),'atol':MAX_ERROR,'mean_tolerance':MEAN_ERROR}
            insert=[e for e in events if e['kind']=='inserted' and e['positions'].numel()>1]
            covered={int(p) for e in insert for p in e['positions'].flatten()}
            checks['insertion_coverage']={'passed':covered==set(range(len(case['input_ids'])))}
            checks['inserted']=[measure(e['embeddings'],expected['embeddings'][0,e['positions'].long()],0.05,0.03) for e in insert]
            pads=[e for e in events if e['kind']=='padding']
            checks['offsets']={'passed':bool(pads) and pads[-1]['offsets']==[[list(x)] for x in proc['offsets']]}
            padding=pads[-1]
            image_positions={i for first,last in proc['offsets'] for i in range(first,last+1)}
            checks['hash_padding']={'passed':padding['original']==case['input_ids'] and len(padding['padded'])==len(case['input_ids']) and all(padding['padded'][i]==token for i,token in enumerate(case['input_ids']) if i not in image_positions) and all(len(set(padding['padded'][first:last+1]))==1 and padding['padded'][first]!=250062 for first,last in proc['offsets'])}
            native_embeddings=expected['embeddings'].clone()
            native_embeddings[expected['input_ids']==250062]=torch.cat([v['projected'] for v in vision]).to(native_embeddings.dtype)
            checks['native_insertion']=[measure(e['embeddings'],native_embeddings[0,e['positions'].long()],0) for e in insert]
            def passed(v):return all(passed(x) for x in v) if isinstance(v,list) else v['passed']
            row['passed']=all(passed(v) for v in checks.values());results.append(row)
            args.directory.joinpath('native-report.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(row),flush=True)
    from k2hv.gate_provenance import capture as provenance
    metadata=provenance()
    metadata['peak_allocated_bytes']=max(e.get('peak_allocated_bytes',0) for e in traces(args.trace).values())
    args.directory.joinpath('native-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    malformed=[]
    for ids,media in (([42,43],cases[0]['image_data']),(cases[0]['input_ids'],[]),(cases[-1]['input_ids'],cases[0]['image_data'])):
        try:
            post(args.url,'/generate',{'input_ids':ids,'image_data':media,'sampling_params':{'max_new_tokens':1}})
            malformed.append({'passed':False,'reason':'malformed image count accepted'})
        except HTTPError as error:
            detail=error.read().decode()
            malformed.append({'passed':error.code in (400,422),'http_status':error.code,'error':detail[:500]})
    args.directory.joinpath('malformed-report.json').write_text(json.dumps(malformed,indent=2)+'\n')
    if not all(r['passed'] for r in results+malformed):raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['capture','evaluate'])
    parser.add_argument('--checkpoint',default='hf/K2-Horizon-V-init')
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--trace',type=Path,default=Path('outputs/step04/native-trace'))
    parser.add_argument('--url',default='http://127.0.0.1:30000')
    args=parser.parse_args();args.directory.mkdir(parents=True,exist_ok=True)
    (capture if args.stage=='capture' else evaluate)(args)


if __name__=='__main__':main()
