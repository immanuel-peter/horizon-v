"""Unpadded packing and gradient parity through pinned Miles batching and SFT loss."""
import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import AutoModelForImageTextToText, AutoProcessor

from .gate_training import fixtures
from .gpu_gates import compare


def initialize():
    from miles.utils.distributed_utils import init_gloo_group
    from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
    from miles.backends.training_utils.parallel import set_parallel_state
    if dist.is_initialized():return
    torch.cuda.set_device(0)
    with tempfile.NamedTemporaryFile() as file:
        path=file.name
    dist.init_process_group('nccl',init_method='file://'+path,rank=0,world_size=1)
    init_gloo_group()
    set_parallel_state(create_fsdp_parallel_state(SimpleNamespace(dp_replicate_size=1)))


def batch(samples,pad_multiplier=128):
    from miles.backends.training_utils.data.rollout import DataIterator,get_batch
    data={'tokens':[torch.tensor(s.tokens,device='cuda') for s in samples],
          'total_lengths':[len(s.tokens) for s in samples],
          'response_lengths':[s.response_length for s in samples],
          'loss_masks':[torch.tensor(s.loss_mask,device='cuda') for s in samples],
          'multimodal_train_inputs':[{k:v.cuda() for k,v in s.multimodal_train_inputs.items()} for s in samples]}
    return get_batch(DataIterator(data,micro_batch_size=len(samples)),list(data),pad_multiplier=pad_multiplier,get_position_ids=True)


def objective(model,samples,pad_multiplier=128):
    from miles.backends.fsdp_utils.actor import FSDPTrainRayActor
    from miles.backends.fsdp_utils.adaptations.specs import resolve_arch_adapter
    from miles.backends.training_utils.loss.hub.losses import sft_loss_function
    b=batch(samples,pad_multiplier=pad_multiplier)
    actor=SimpleNamespace(arch_adapter=resolve_arch_adapter(model.config))
    inputs=FSDPTrainRayActor._get_model_inputs_args(actor,b)
    args=SimpleNamespace(qkv_format='thd',allgather_cp=False,true_on_policy_mode=True,rollout_temperature=1.0,log_probs_backend='torch',log_probs_chunk_size=0,debug_unified_grad_fused_logprob=False)
    logits=model(**inputs).logits
    return sft_loss_function(args,b,logits,lambda v:v.float().sum())[0],sum(s.response_length for s in samples),b


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    from .gpu_numerics import enable
    enable();initialize();torch.manual_seed(0)
    p=AutoProcessor.from_pretrained('hf/K2-Horizon-V-init',trust_remote_code=True,local_files_only=True)
    m=AutoModelForImageTextToText.from_pretrained('hf/K2-Horizon-V-init',trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    from miles.backends.fsdp_utils.adaptations.precision import apply_fp32_master
    from miles.backends.fsdp_utils.actor import apply_fsdp2,FSDPTrainRayActor
    from miles.backends.training_utils.parallel import get_parallel_state
    from miles.backends.fsdp_utils.adaptations.specs import resolve_arch_adapter
    resolve_arch_adapter(m.config).patch_model(m,SimpleNamespace())
    m=apply_fp32_master(m)
    full_state=m.state_dict()
    m=apply_fsdp2(m,mesh=get_parallel_state().get_mesh('fsdp'),args=SimpleNamespace(fp16=False),param_dtype=torch.bfloat16,reduce_dtype=torch.float32)
    m=FSDPTrainRayActor._fsdp2_load_full_state_dict(SimpleNamespace(),m,full_state,get_parallel_state().get_mesh('fsdp'),None)
    del full_state
    s=fixtures(p);pair=[s[0],s[-1]];total=sum(x.response_length for x in pair)
    separate=0.0
    for x in pair:
        value,count,_=objective(m,[x],pad_multiplier=1);separate+=value.detach().item()/total;(value/total).backward()
    gradients={n:(v.grad.to_local() if hasattr(v.grad,'to_local') else v.grad).detach().clone() for n,v in m.projector.named_parameters()};m.zero_grad(set_to_none=True)
    value,count,b=objective(m,pair);(value/count).backward()
    from .gate_provenance import capture as provenance
    report={'provenance':provenance(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'separate_loss':separate,'packed_loss':(value/count).item(),'loss_atol':0.02,'supervised_tokens':total,'packed_length':b['tokens'].shape[1],'cu_seqlens':b['cu_seqlens_host'],'packed_pad_multiplier':128,'individual_pad_multiplier':1,'position_resets':(b['position_ids'][0]==0).nonzero().flatten().tolist(),'source':'Miles get_batch -> real FSDP2 BF16 compute/FP32 master and reduction -> FSDP actor input method -> sft_loss_function; unpadded individual forwards','gradients':{},'passed':False}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    try:
        assert abs(report['packed_loss']-separate)<=0.02, report
        report['gradients']={n:compare(v.grad.to_local() if hasattr(v.grad,'to_local') else v.grad,gradients[n],atol=0.005,rtol=0.03) for n,v in m.projector.named_parameters()}
        assert all(v.grad is None for n,v in m.named_parameters() if not n.startswith('projector.'))
        report['passed']=True
    finally:
        args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        dist.destroy_process_group()


if __name__=='__main__':main()
