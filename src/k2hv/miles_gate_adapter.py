"""Minimal pinned-Miles FSDP adapter and opt-in evidence for the small SFT gate."""
import json
import os
from pathlib import Path

import torch
from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter


class K2HorizonVAdapter(ArchAdapter):
    model_types=frozenset({'k2_horizon_v'})

    def packing_kwargs(self, *, cu_seqlens, cu_seqlens_host, max_seqlen):
        return {"k2hv_cu_seqlens": tuple(cu_seqlens_host)}

    def patch_model(self,model,args):
        from .gpu_numerics import enable
        enable()
        model.freeze_pretrained_modules()
        from .packed_attention import install
        install(model)
        from torch.distributed.fsdp import MixedPrecisionPolicy
        model.projector.float()
        model.projector._k2hv_mixed_precision = MixedPrecisionPolicy(param_dtype=torch.float32,reduce_dtype=torch.float32,cast_forward_inputs=False)
        model._no_split_modules = list(model._no_split_modules) + ['Projector']
        directory=os.environ.get('K2HV_MILES_GATE')
        if not directory:return
        root=Path(directory);root.mkdir(parents=True,exist_ok=True)
        import sys,sglang
        from k2hv_sglang.check_runtime import check_source
        source=Path(sglang.__file__).parent/'srt'
        verified=check_source(source)
        (root/'worker-runtime.json').write_text(json.dumps({'python':sys.executable,'sglang_source':str(source),'verified_source_files':verified,'provenance':__import__('k2hv.gate_provenance',fromlist=['capture']).capture()},indent=2))
        names=[n for n,p in model.named_parameters() if p.requires_grad]
        assert names and all(n.startswith('projector.') for n in names), names
        (root/'trainable.json').write_text(json.dumps(names,indent=2))
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.hf_checkpoint,trust_remote_code=True,local_files_only=True)
        close=tokenizer.encode('</ifm|think>',add_special_tokens=False)
        end=tokenizer.convert_tokens_to_ids('<|ifm|im_end|>')
        def forward(module,inputs,kwargs,output):
            if not module.training or not torch.is_grad_enabled():return
            (root/'memory.json').write_text(json.dumps({'peak_allocated_bytes':torch.cuda.max_memory_allocated()}))
            ids=kwargs['input_ids'][0].detach().tolist()
            logits=output.logits.detach()[0].float()
            spans=[]
            for i in range(len(ids)-len(close)+1):
                if ids[i:i+len(close)]!=close:continue
                first=i+len(close)
                last=ids.index(end,first)+1
                labels=torch.tensor(ids[first:last],device=logits.device)
                value=torch.nn.functional.cross_entropy(logits[first-1:last-1],labels).item()
                spans.append({'loss':value,'tokens':ids[first:last],'decoded':tokenizer.decode(ids[first:last],skip_special_tokens=False)})
            assert spans, 'Miles batch has no supervised answer spans'
            with (root/'forward.jsonl').open('a') as file:file.write(json.dumps(spans)+'\n')
        def backward(module,grad_input,grad_output):
            gradients=[g for g in grad_output if g is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            norm=sum(g.float().square().sum() for g in gradients).sqrt().item()
            assert norm>0, 'Frozen LM did not pass gradients to projector outputs'
            with (root/'backward.jsonl').open('a') as file:file.write(json.dumps({'image_feature_gradient_norm':norm})+'\n')
        model.register_forward_hook(forward,with_kwargs=True)
        model.projector.register_full_backward_hook(backward)
