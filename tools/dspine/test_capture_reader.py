import json
from pathlib import Path
import struct
import tempfile
import unittest
try:
    import capture_reader
except ModuleNotFoundError:
    capture_reader=None


@unittest.skipUnless(capture_reader is not None,"PyTorch required")
class CaptureReaderTest(unittest.TestCase):
    def fixture(self, directory, tokens):
        p=Path(directory)
        (p/'metadata.json').write_text(json.dumps({'schema':'saber-target-aux/1','dtype':'BF16','byte_order':'little','row_bytes':2}))
        (p/'test.events.jsonl').write_text(json.dumps({'position':0,'rows':len(tokens)})+'\n')
        (p/'test.tokens.u32le').write_bytes(struct.pack('<'+'I'*len(tokens),*tokens))
        (p/'test.aux.bf16').write_bytes(b'\0\0'*len(tokens))
        return {'choices':[{'finish_reason':'length','token_ids':[3,4,5]}], 'prompt_token_ids':[1,2], 'usage':{'prompt_tokens':2,'completion_tokens':3}}

    def test_terminal_internal_work_is_trimmed_to_actual_api_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            response=self.fixture(d,[1,2,3,4,5,6,7])
            result=capture_reader.load_capture(d,'test',response)
            self.assertEqual(result['features'].shape,(5,1))
            self.assertEqual(result['proof']['trimmed_terminal_rows'],2)
            self.assertEqual(result['tokens'].tolist(),[1,2,3,4,5])

    def test_final_generated_token_need_not_have_been_fed_back(self):
        with tempfile.TemporaryDirectory() as d:
            response=self.fixture(d,[1,2,3,4])
            self.assertEqual(capture_reader.load_capture(d,'test',response)['features'].shape,(4,1))

    def test_wrong_prefix_or_journal_gap_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            response=self.fixture(d,[1,2,9,4,5])
            with self.assertRaisesRegex(ValueError,'prefix'):
                capture_reader.load_capture(d,'test',response)
            self.fixture(d,[1,2,3,4,5])
            (Path(d)/'test.events.jsonl').write_text(json.dumps({'position':1,'rows':5})+'\n')
            with self.assertRaisesRegex(ValueError,'Discontinuous'):
                capture_reader.load_capture(d,'test',response)

    def test_missing_internal_rows_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            response=self.fixture(d,[1,2,3])
            with self.assertRaisesRegex(ValueError,'terminal range'):
                capture_reader.load_capture(d,'test',response)


if __name__=='__main__':unittest.main()
