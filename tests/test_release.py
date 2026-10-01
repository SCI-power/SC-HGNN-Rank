from pathlib import Path
import json
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from recompute_available_external_scores import available_mean
from package_splits import build_split_configs


class ReleaseTests(unittest.TestCase):
    def test_measured_zero_and_missing_sources(self):
        values = np.array([[0., .6, .9, .9], [1., .2, .3, .4]])
        present = np.array([[True, True, False, False], [False, False, False, False]])
        result, counts = available_mean(values, present)
        self.assertAlmostEqual(result[0], .3)
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(counts.tolist(), [2, 0])

    def test_invalid_evaluable_value(self):
        with self.assertRaises(ValueError):
            available_mean(np.array([[np.nan]]), np.array([[True]]))

    def test_fixed_folds(self):
        labels = pd.read_csv(ROOT / 'data/model/Table_HGNN3_axis_label_table.csv')
        self.assertEqual(len(labels), 600)
        splits = build_split_configs(labels)
        self.assertEqual(len(splits), 13)
        for family, group in splits.groupby('validation_type'):
            indices = [int(v) for value in group.test_indices for v in value.split(';')]
            self.assertEqual(sorted(indices), list(range(600)), family)

    def test_checkpoint_coverage(self):
        rows = json.loads((ROOT / 'checkpoints/index.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows), 65)
        self.assertEqual(len({(r['fold'], r['seed']) for r in rows}), 65)
        self.assertTrue(all((ROOT / r['path']).is_file() for r in rows))

    def test_selected_priority_values(self):
        table = pd.read_csv(ROOT / 'data/supporting/priority_scores/retained_24_priority_scores.csv').set_index('frozen_axis_id')
        expected = {'COAD_AXIS_034': .807259981236907, 'LIHC_AXIS_092': .7106715907446779,
                    'STAD_AXIS_021': .7680616281418959}
        for key, value in expected.items():
            self.assertAlmostEqual(float(table.loc[key, 'composite_priority_score']), value, places=10)


if __name__ == '__main__':
    unittest.main()
