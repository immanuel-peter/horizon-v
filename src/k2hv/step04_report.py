"""Fail closed when assembling the independently executed step 04 gate evidence."""
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .gate_provenance import capture


def assemble(root):
    current=capture()
    results=[]
    artifacts={}
    def load(name):
        path=root/name
        artifacts[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        return json.loads(path.read_text())
    def identity(metadata):
        hashes=metadata.get('source_sha256',metadata.get('code_sha256',{}))
        assert hashes,'Missing executed source hashes'
        for name,digest in hashes.items():
            assert current['source_sha256'].get(name)==digest,f'Stale executed source: {name}'
    def gate(name,run):
        try:
            evidence=run()
            results.append({'gate':name,'status':'passed','evidence':evidence})
        except FileNotFoundError as error:
            results.append({'gate':name,'status':'blocked','error':str(error)})
        except Exception as error:
            results.append({'gate':name,'status':'failed','error':str(error)})
    def hf(name):
        report=load('hf.json');identity(report)
        row=next(r for r in report['results'] if r['gate']==name)
        assert row['status']=='passed',row
        return row
    for name in ('cuda','load_hf','text_hf','vision_hf'):
        gate(name,lambda name=name:hf(name))
    def native_load():
        runtime=load('runtime.json');smoke=load('smoke.json')
        assert runtime['verified_source_files']>=15 and runtime['reasoning_parser']=='k2_horizon'
        assert smoke['hf_image_tokens']==256 and smoke['miles_image']['meta_info']['prompt_tokens']==smoke['hf_prompt_tokens']
        return {'runtime':runtime,'smoke_requests':['text','Miles input_ids + image_data','chat image_url']}
    gate('load_sglang',native_load)
    def text():
        stages=[]
        for name in ('text-chunked.json','text-unchunked.json'):
            report=load(name);identity(report['provenance'])
            assert len(report['results'])==6 and all(r['status']=='passed' for r in report['results']),report['results']
            assert report['max_error_tolerance']==0.2 and report['mean_error_tolerance']==0.03
            stages.append({'path':name,'results':report['results']})
        identity(load('text-reference.json')['provenance'])
        return stages
    gate('text_sglang',text)
    def native():
        identity(load('native/hf-provenance.json'));identity(load('native/native-provenance.json'))
        rows=load('native/native-report.json');malformed=load('native/malformed-report.json')
        assert len(rows)==12 and all(r['passed'] for r in rows)
        assert len(malformed)==3 and all(r['passed'] for r in malformed)
        return {'cases':rows,'malformed':malformed}
    gate('native_vision_preprocessing_tokens_insertion_logprobs',native)
    def simple(name):
        report=load(name);identity(report['provenance']);assert report['passed'],report
        return {k:v for k,v in report.items() if k!='provenance'}
    gate('cache_projector_update_fresh_engine',lambda:simple('cache/report.json'))
    gate('packing_unpadded_and_gradients',lambda:simple('packing.json'))
    def overfit():
        report=simple('overfit.json');training=load('training-summary.json')
        identity(training['worker']['provenance'])
        assert len(report['examples'])==64 and all(r['loss']<0.05 and r['teacher_forced_exact'] and r['free_generation_exact'] for r in report['examples'])
        assert report['frozen_tensors_checked']==666 and report['frozen_unchanged'] and report['projector_changed']
        assert len(training['trainable'])==6 and all(n.startswith('projector.') for n in training['trainable'])
        assert training['finite_nonzero_gradient_steps']>0 and training['actual_miles_optimizer_updates']>=512
        return {'evaluation':report,'training':training}
    gate('miles_64_example_overfit',overfit)
    def regression():
        path=root/'regression.xml';tree=ET.parse(path)
        suites=list(tree.getroot().iter('testsuite'))
        counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        assert counts['tests']>=160 and counts['failures']==counts['errors']==counts['skipped']==0,counts
        artifacts['regression.xml']={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        return counts
    gate('cpu_regression',regression)
    return {'schema':2,'step04_complete':all(r['status']=='passed' for r in results),'provenance':current,'results':results,'artifacts':artifacts}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/step04/final'));p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    report=assemble(a.root);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
    for row in report['results']:print(f"{row['gate']}: {row['status']} {row.get('error','')}")
    if not report['step04_complete']:raise SystemExit(1)


if __name__=='__main__':main()
