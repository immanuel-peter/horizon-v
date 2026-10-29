"""Extract optimizer, mask and gradient evidence from the actual Miles SFT run."""
import argparse
import ast
import json
import math
import re
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--logs',nargs='+',type=Path,required=True);p.add_argument('--directory',type=Path,default=Path('outputs/step04/miles-sft'));p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    curve={}
    for path in a.logs:
        for line in path.read_text().splitlines():
            match=re.search(r"step (\d+): (\{.*\})",line)
            if match:
                values=ast.literal_eval(match[2])
                if 'train/loss' in values:curve[int(match[1])]=values
    assert len(curve)>=512,len(curve)
    assert all(math.isfinite(v['train/loss']) and math.isfinite(v['train/grad_norm']) for v in curve.values())
    gradients=[json.loads(line) for line in (a.directory/'backward.jsonl').read_text().splitlines()]
    assert gradients and all(math.isfinite(g['image_feature_gradient_norm']) and g['image_feature_gradient_norm']>0 for g in gradients)
    forward=[json.loads(line) for line in (a.directory/'forward.jsonl').read_text().splitlines()]
    spans=sorted({s['decoded'] for row in forward for s in row})
    assert len(spans)==16 and all(s.endswith('<|ifm|im_end|>') and 'think' not in s and 'image' not in s for s in spans)
    report={'worker':json.loads((a.directory/'worker-runtime.json').read_text()),'trainable':json.loads((a.directory/'trainable.json').read_text()),'memory':json.loads((a.directory/'memory.json').read_text()),'actual_miles_optimizer_updates':len(curve),'finite_nonzero_gradient_steps':sum(v['train/grad_norm']>0 for v in curve.values()),'frozen_lm_passed_gradients_steps':len(gradients),'image_feature_gradient_norm_range':[min(g['image_feature_gradient_norm'] for g in gradients),max(g['image_feature_gradient_norm'] for g in gradients)],'supervised_spans':spans,'loss_curve':[{'step':step,**values} for step,values in sorted(curve.items())]}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('worker','loss_curve')},indent=2))


if __name__=='__main__':main()
