"""Measure BF16 prefix invariance before and after the selected numerical policy."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText,AutoTokenizer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',default='hf/K2-Horizon-V-init');p.add_argument('--report',type=Path,required=True);a=p.parse_args();torch.manual_seed(0)
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,trust_remote_code=True,local_files_only=True)
    ids=tokenizer('Explain a sequence of integers. '*128,add_special_tokens=False,return_tensors='pt')['input_ids'].cuda()
    model=AutoModelForImageTextToText.from_pretrained(a.checkpoint,trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    report={}
    for phase in ('stock','batch_invariant_math'):
        if phase!='stock':
            from .gpu_numerics import enable
            enable()
        rows=[]
        with torch.no_grad():
            for n in (14,256,769):
                logits=model(input_ids=ids[:,:n],use_cache=False).logits[0,:-1].float()
                values=logits.log_softmax(-1).gather(-1,ids[0,1:n,None]).flatten().cpu()
                rows.append({'input_length':n,'early_logprobs':values[:13].tolist()})
        report[phase]=rows
    from .gate_provenance import capture
    report['provenance']=capture();report['input_ids']=ids.cpu().tolist();a.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
