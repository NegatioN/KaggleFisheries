import unittest
import annotationHandler as ah

big_size = (500, 500)
small_size = (224, 224)


# bb_params = ['height', 'width', 'x', 'y']

class AnnotationHandlerTestCase(unittest.TestCase):
    def test_convert_bb_size_scales_down_quadratic(self):
        bb = [50, 50, 50, 50]
        scaled_down_bb = ah.convert_bb_size(bb, small_size, big_size)

        self.assertEqual([22, 22, 22, 22], scaled_down_bb)

    def test_convert_bb_size_scales_down_complex(self):
        bb = [20, 50, 33, 77]
        scaled_down_bb = ah.convert_bb_size(bb, small_size, big_size)
        self.assertEqual([8, 22, 14, 34], scaled_down_bb)

    def test_convert_bb_size_scales_up(self):
        bb = [20, 50, 33, 77]
        scaled_down_bb = ah.convert_bb_size(bb, big_size, small_size)
        self.assertEqual([44, 111, 73, 171], scaled_down_bb)


if __name__ == '__main__':
    unittest.main()
