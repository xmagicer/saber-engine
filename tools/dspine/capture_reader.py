"""Validate committed target captures against their actual API token streams."""
import json
from pathlib import Path
import torch


def load_capture(directory, stem, response, max_extra_rows=7):
    directory=Path(directory)
    meta=json.loads((directory/'metadata.json').read_text())
    if meta['schema']!='saber-target-aux/1' or meta['dtype']!='BF16' or meta['byte_order']!='little':
        raise ValueError('Unsupported target capture format')
    events=[json.loads(line) for line in (directory/(stem+'.events.jsonl')).read_text().splitlines()]
    cursor=0
    for event in events:
        if event['position']!=cursor or event['rows']<=0:raise ValueError('Discontinuous target capture')
        cursor+=event['rows']
    token_file=directory/(stem+'.tokens.u32le');aux_file=directory/(stem+'.aux.bf16')
    if token_file.stat().st_size!=cursor*4 or aux_file.stat().st_size!=cursor*meta['row_bytes']:
        raise ValueError('Target capture payload length disagrees with journal')
    if response['choices'][0]['finish_reason']=='abort':raise ValueError('Aborted source generation')
    prompt=response['prompt_token_ids'];generated=response['choices'][0]['token_ids']
    if len(prompt)!=response['usage']['prompt_tokens'] or len(generated)!=response['usage']['completion_tokens']:
        raise ValueError('Source API token counts disagree')
    expected=torch.tensor(prompt+generated,dtype=torch.int64)
    tokens=torch.from_file(str(token_file),shared=False,size=cursor,dtype=torch.int32).long()
    common=min(cursor,expected.numel())
    if cursor<expected.numel()-1 or cursor>expected.numel()+max_extra_rows:
        raise ValueError('Unexpected target capture terminal range')
    if not torch.equal(tokens[:common],expected[:common]):raise ValueError('Captured input tokens differ from API prefix')
    features=torch.from_file(str(aux_file),shared=False,size=cursor*(meta['row_bytes']//2),dtype=torch.bfloat16).reshape(cursor,-1)
    # Never expose internal terminal work beyond the caller-visible token budget to a replay.
    proof={'stem':stem,'prompt_tokens':len(prompt),'generated_tokens':len(generated),
           'captured_rows':cursor,'usable_rows':common,'trimmed_terminal_rows':max(0,cursor-common),
           'exact_input_token_match':True,'contiguous':True}
    return {'metadata':meta,'tokens':expected,'features':features[:common],
            'prompt_tokens':len(prompt),'proof':proof}
