import unittest
import target_layers as Layers


class TargetLayerMappingTests(unittest.TestCase):
    def test_official_post_decoder_ids_map_to_next_saber_input(self):
        ids=[1,16,31,46,61];expected=[2,17,32,47,62]
        self.assertEqual(Layers.capture_layers_from_dspine(ids),expected)
        self.assertEqual(Layers.validate_dspine_capture({'target_layer_ids':expected},ids),expected)
        with self.assertRaises(ValueError):Layers.validate_dspine_capture({'target_layer_ids':ids},ids)
        self.assertEqual(Layers.capture_layers_from_dspine([0]),[1])

    def test_malformed_decoder_id_lists_are_refused(self):
        for ids in [[],[-1],[1,1],[2,1],[False],[1.0]]:
            with self.assertRaises(ValueError):Layers.capture_layers_from_dspine(ids)


if __name__=='__main__':unittest.main()
