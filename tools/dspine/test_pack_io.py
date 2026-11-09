import json
import tempfile
from pathlib import Path
import unittest
try:
    import torch
    import pack_io as Pack
except ModuleNotFoundError:
    torch=None


@unittest.skipUnless(torch is not None,'PyTorch required')
class PackTests(unittest.TestCase):
    def test_fp8_decode_matches_documented_values_and_row_scales(self):
        w=torch.tensor([[0,1,0x38],[0x40,0x7e,0xb8]],dtype=torch.uint8).view(torch.float8_e4m3fn)
        actual=Pack.dequantize_channel_fp8(w,torch.tensor([2.,.5]))
        expected=torch.tensor([[0,1/256,2],[1,224,-.5]],dtype=torch.bfloat16)
        self.assertTrue(torch.equal(actual,expected))

    def test_embedding_rank_offsets_and_head_channels_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);ranks=[]
            for rank in range(2):
                embed=torch.tensor([[1+rank*4,2+rank*4],[3+rank*4,4+rank*4]],dtype=torch.bfloat16)
                data=bytes(4096)+embed.view(torch.uint8).numpy().tobytes()+bytes([0x38]*4)+torch.tensor([1.+rank*2,2.+rank*2]).numpy().tobytes()
                (directory/f'rank{rank}.pack').write_bytes(data)
                records=[{'name':'model.language_model.embed_tokens.weight','shape':[2,2],'data_off':4096,'length':8,'layout_id':0},
                         {'name':'lm_head.w8','shape':[2,2],'data_off':4104,'length':4,'layout_id':15},
                         {'name':'lm_head.ws','shape':[2],'data_off':4108,'length':8,'layout_id':16}]
                ranks.append({'rank':rank,'file':f'rank{rank}.pack','file_bytes':len(data),'records':records})
            (directory/'manifest.json').write_text(json.dumps({'magic':'SBRPACK1','format_version':2,'ranks':ranks}))
            target=Pack.TargetPack(directory)
            self.assertTrue(torch.equal(target.embedding_rows(torch.tensor([3,0,2])),torch.tensor([[7,8],[1,2],[5,6]],dtype=torch.bfloat16)))
            self.assertTrue(torch.equal(target.load_head(chunk=1),torch.tensor([[1,1],[2,2],[3,3],[4,4]],dtype=torch.bfloat16)))
            with self.assertRaises(ValueError):target.embedding_rows(torch.tensor([-1]))
            with self.assertRaises(ValueError):target.embedding_rows(torch.tensor([4]))


if __name__=='__main__':unittest.main()
