"""Read-only target embedding/head views from verified SBRPACK1 TP shards."""
import json
import math
from pathlib import Path
import torch


def dequantize_channel_fp8(weight, scale):
    """Match Saber's bf16(f32(E4M3FN weight) * f32(channel scale))."""
    if weight.dtype!=torch.float8_e4m3fn or scale.dtype!=torch.float32 or scale.shape!=(weight.shape[0],):
        raise ValueError('Expected row-major E4M3FN weights and FP32 row scales')
    return (weight.float()*scale[:,None]).to(torch.bfloat16)


class TargetPack:
    """CPU memory maps; embedding lookups transfer only the selected rows."""
    def __init__(self,directory):
        self.directory=Path(directory)
        manifest=json.loads((self.directory/'manifest.json').read_text())
        if manifest['magic']!='SBRPACK1' or manifest['format_version']!=2:
            raise ValueError('Unsupported target pack')
        self.shards=[];self.vocab=0;self.hidden=None
        for shard in sorted(manifest['ranks'],key=lambda s:s['rank']):
            if shard['file']!=f"rank{shard['rank']}.pack":
                raise ValueError('Unexpected shard filename')
            path=self.directory/shard['file']
            if path.stat().st_size!=shard['file_bytes']:
                raise ValueError('Target pack size mismatch')
            mapped=torch.from_file(str(path),shared=False,size=shard['file_bytes'],dtype=torch.uint8)
            records={x['name']:x for x in shard['records']}
            embed=self.record(mapped,records['model.language_model.embed_tokens.weight'],torch.bfloat16,0)
            head=self.record(mapped,records['lm_head.w8'],torch.float8_e4m3fn,15)
            scale=self.record(mapped,records['lm_head.ws'],torch.float32,16)
            if embed.shape!=head.shape or scale.shape!=(head.shape[0],):
                raise ValueError('Target embedding/head shape mismatch')
            if self.hidden is not None and embed.shape[1]!=self.hidden:
                raise ValueError('Target shard hidden dimensions differ')
            self.hidden=embed.shape[1]
            self.shards.append({'base':self.vocab,'rows':embed.shape[0],'embedding':embed,'head':head,'scale':scale,'mapped':mapped})
            self.vocab+=embed.shape[0]

    @staticmethod
    def record(mapped,entry,dtype,layout):
        count=math.prod(entry['shape']);size=torch.empty((),dtype=dtype).element_size()
        off,length=entry['data_off'],entry['length']
        if entry['layout_id']!=layout or length!=count*size or off<0 or off+length>mapped.numel():
            raise ValueError('Unsupported target record layout or range: '+entry['name'])
        return mapped[off:off+length].view(dtype).reshape(entry['shape'])

    @torch.no_grad()
    def embedding_rows(self,ids):
        ids=ids.to(device='cpu',dtype=torch.int64)
        if ids.ndim!=1 or bool(((ids<0)|(ids>=self.vocab)).any()):
            raise ValueError('Embedding ids outside vocabulary')
        output=torch.empty((ids.numel(),self.hidden),dtype=torch.bfloat16)
        for shard in self.shards:
            mask=(ids>=shard['base'])&(ids<shard['base']+shard['rows'])
            if bool(mask.any()):output[mask]=shard['embedding'][ids[mask]-shard['base']]
        return output

    @torch.no_grad()
    def load_head(self,device='cpu',chunk=4096):
        if chunk<1:raise ValueError('chunk must be positive')
        head=torch.empty((self.vocab,self.hidden),dtype=torch.bfloat16,device=device)
        for shard in self.shards:
            for start in range(0,shard['rows'],chunk):
                end=min(start+chunk,shard['rows'])
                values=dequantize_channel_fp8(shard['head'][start:end],shard['scale'][start:end])
                head[shard['base']+start:shard['base']+end]=values.to(device)
        return head
