import unittest
import utils
from pandas import read_csv

test_data = read_csv('test_data.csv')


class UtilsTest(unittest.TestCase):
    def test_filter_wild_pitches(self):
        lengths = (34521, 43197, 44652, 44798)

        for i in range(3, 6):
            result = len(utils.filter_wild_pitches(test_data, i))
            self.assertEqual(result, lengths[i - 3])

    def test_get_pitch_count(self):
        pitchers = {541650: 91, 571035: 459, 657681: 374,
                    571710: 4363, 573244: 344, 573204: 89}

        for pitcher in pitchers.keys():
            result = utils._get_pitch_count(test_data, pitcher)
            self.assertEqual(result, pitchers[pitcher])

    def test_filter_pitches(self):
        pitches_to_remove = ['KN', 'IN', 'FO', 'EP', 'PO', 'SC', 'UN', 'FA', 'AB', 'FF', 'FT']
        result = utils.filter_pitches(test_data, pitches_to_remove)

        for pitch in pitches_to_remove:
            self.assertIs(pitch in result.pitch_type.values, False)


if __name__ == '__main__':
    unittest.main()
