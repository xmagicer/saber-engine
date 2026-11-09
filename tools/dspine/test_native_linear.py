import unittest
try:
    import torch
    import native_linear
except ModuleNotFoundError:
    torch=None


@unittest.skipUnless(torch is not None,'PyTorch required')
class NativeLinearPackingTests(unittest.TestCase):
    def test_preshuffle_matches_explicit_native_addresses(self):
        n,k=32,128
        x=torch.arange(n*k).remainder(251).to(torch.uint8).reshape(n,k)
        out=native_linear.preshuffle(x)
        for row in range(n):
            for col in range(k):
                offset=(col//32)*512+((col%32)//16)*256+(row%16)*16+col%16
                self.assertEqual(int(out[row//16,offset]),int(x[row,col]))

    def test_quantized_blocks_have_independent_scales_and_finite_zeros(self):
        x=torch.zeros((256,256),dtype=torch.bfloat16)
        x[:128,:128]=1
        x[128:,128:]=-64
        original=x.clone()
        packed,scales=native_linear.quantize_weight(x)
        self.assertEqual(packed.shape,(16,4096))
        self.assertEqual(scales.shape,(2,2))
        self.assertTrue(torch.equal(x,original))
        self.assertTrue(torch.isfinite(scales).all())
        self.assertAlmostEqual(float(scales[1,1]/scales[0,0]),64,places=4)
        with self.assertRaises(ValueError):native_linear.quantize_weight(x[:129])

    def test_channel_weights_round_trip_row_scales_without_changing_source(self):
        x=torch.zeros((32,128),dtype=torch.bfloat16);x[1]=1;x[2]=-64
        before=x.clone();q,scales=native_linear.quantize_channel_weight(x)
        decoded=(q.view(torch.float8_e4m3fn).float()*scales[:,None]).bfloat16()
        self.assertTrue(torch.equal(decoded,x))
        self.assertTrue(torch.equal(before,x))
        self.assertTrue(torch.isfinite(scales).all())
        self.assertEqual(q.shape,x.shape)


if __name__=='__main__':unittest.main()
