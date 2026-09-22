import unittest
import numpy as np
from src.parcel_segmenter import ParcelSegmenter

class TestParcelSegmentation(unittest.TestCase):
    def setUp(self):
        self.config = {
            'spatial': {
                'resolution_meters': 10.0,
                'crs': 'EPSG:32650',
                'utm_zone': 50
            },
            'segmentation': {
                'apply_boundary_erosion': True,
                'fill_internal_holes': True,
                'smooth_boundaries': True,
                'subdivide_oversized': True,
                'min_parcel_area_m2': 200.0,
                'max_parcel_area_m2': 5000.0,
                'discard_oversized': False,
                'max_export_parcels': 100,
                'connectivity': 8
            }
        }

    def test_oversized_cluster_subdivision(self):
        segmenter = ParcelSegmenter(self.config)
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[10:45, 10:45] = 1
        mask[55:90, 55:90] = 1
        mask[40:60, 40:60] = 1
        parcel_mask, metadata = segmenter.segment_parcels(mask)
        self.assertGreater(len(metadata), 1, 'Oversized cluster should be subdivided')
        total_extracted_m2 = sum(p['area_m2'] for p in metadata)
        raw_crop_m2 = np.sum(mask > 0) * 100.0
        self.assertGreater(total_extracted_m2, raw_crop_m2 * 0.85, 'Most cropland area should be preserved')

    def test_single_homogeneous_large_parcel_preserved(self):
        segmenter = ParcelSegmenter(self.config)
        mask = np.zeros((80, 80), dtype=np.int32)
        mask[20:60, 20:60] = 1
        parcel_mask, metadata = segmenter.segment_parcels(mask)
        self.assertGreaterEqual(len(metadata), 1, 'Large homogeneous parcel should be kept')
        self.assertGreater(metadata[0]['area_m2'], 5000.0, 'Large parcel area should be intact')

    def test_noise_filtering(self):
        segmenter = ParcelSegmenter(self.config)
        mask = np.zeros((50, 50), dtype=np.int32)
        mask[10, 10] = 1
        mask[20:35, 20:35] = 1
        parcel_mask, metadata = segmenter.segment_parcels(mask)
        self.assertEqual(parcel_mask[10, 10], 0, 'Isolated 1-pixel noise should be filtered')
        self.assertEqual(len(metadata), 1, 'Only valid plot should be extracted')

if __name__ == '__main__':
    unittest.main()
