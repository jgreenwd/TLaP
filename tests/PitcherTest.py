import unittest
from pandas import read_csv
import Pitcher
import utils

test_data = read_csv('test_data.csv')


class PitcherTest(unittest.TestCase):
    def test_constructor(self):
        pitcher = Pitcher.Pitcher()
        self.assertIsInstance(pitcher, Pitcher.Pitcher)

    def test_pid(self):
        p_id = 123456
        pitcher = Pitcher.Pitcher()
        pitcher.set_p_id(p_id)
        self.assertEqual(pitcher._p_id, 123456)
        self.assertEqual(pitcher.get_p_id(), 123456)

    def test_seasons(self):
        seasons = [2015, 2016, 2017, 2018]
        pitcher = Pitcher.Pitcher()
        pitcher.set_seasons(seasons)
        self.assertListEqual(pitcher._seasons, seasons)
        self.assertListEqual(pitcher.get_seasons(), seasons)

    # test static methods
    def test_get_h_and_c_scores(self):
        scores = ((0.00, 1.00),
                  (0.01, 0.02),
                  (0.25, 0.44),
                  (0.04, 0.04),
                  (0.08, 0.07),
                  (0.41, 0.31),
                  (0.70, 0.53),
                  (0.29, 0.24))

        x, y = utils.preprocess(test_data, accuracy=False, confidence=True)
        start = 0

        for index, end in enumerate(range(3000, 27000, 3000)):
            result = Pitcher.Pitcher.get_h_and_c_scores(x[start:end], y[start:end], index + 1)
            start = end
            self.assertEqual(scores[index], tuple(map(lambda z: round(z, 2), result)))

    def test_estimate_n_clusters(self):
        homo_dict = {2: 0.4470766933289132, 3: 0.47808042883291746,
                     4: 0.5018182271669309, 5: 0.5442452822405283,
                     6: 0.5211664518186989, 7: 0.7170894102690551,
                     8: 0.722867528024443, 9: 0.7164576007108636,
                     10: 0.7162035849643891, 11: 0.6662504414068767}

        comp_dict = {2: 0.9650297257285013, 3: 0.7191888565136932,
                     4: 0.6415045936683161, 5: 0.730688517115183,
                     6: 0.6214551668634116, 7: 0.7106572225068184,
                     8: 0.6756289502990558, 9: 0.6800344545167626,
                     10: 0.6839596096681522, 11: 0.6552245464674255}

        results = [2, 3, 4, 4, 6, 7, 7, 7, 7]

        for i in range(3, 12):
            test_dict1 = {k: v for k, v in homo_dict.items() if k < i}
            test_dict2 = {k: v for k, v in comp_dict.items() if k < i}

            result = Pitcher.Pitcher._estimate_n_clusters(test_dict1, test_dict2)

            self.assertEqual(result, results[i - 3])


if __name__ == '__main__':
    unittest.main()
