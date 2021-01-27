from sklearn.metrics import homogeneity_score, completeness_score
from sklearn.mixture import GaussianMixture


class Pitcher:
    def __init__(self, p_id=0, seasons=None):
        self._p_id = p_id
        self._seasons = seasons

    def set_p_id(self, p_id):
        self._p_id = p_id

    def get_p_id(self):
        return self._p_id

    def set_seasons(self, seasons):
        self._seasons = seasons

    def get_seasons(self):
        return self._seasons

    @staticmethod
    def get_h_and_c_scores(x, y, n_components):
        """ Return homogeneity & completeness for GMM with n_components

            :param x:            (DataFrame) processed pitch data
            :param y:            (Series) encoded target values
            :param n_components: (Integer) number of clusters to test (Default 5)

            :return:             (homogeneity_score, completeness_score)
        """
        _clf = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=500, random_state=42)
        _preds = _clf.fit_predict(x)

        return homogeneity_score(y, _preds), completeness_score(y, _preds)

    @staticmethod
    def _estimate_n_clusters(homogeneity, completeness):
        """ Return estimated number of clusters found in GMM model

            :param homogeneity:  (Dictionary - n:score) of homogeneity scores
            :param completeness: (Dictionary - n:score) of completeness scores

            :return: (integer) estimated of n_clusters
        """
        if homogeneity.keys() != completeness.keys():
            raise ValueError('_estimate_n_clusters: homogeneity & completeness scores must have identical keys')

        _scores = {k: abs(homogeneity[k] - completeness[k]) for k in homogeneity.keys()}

        return min(_scores, key=_scores.get)

    @staticmethod
    def get_n_clusters(df):
        """ Return estimated number of clusters found with GMM model

            :param df: DataFrame of valid pitch data

            :return: (integer) estimated of n_clusters
        """
        from utils import preprocess

        try:
            nmin = 2
            nmax = 8

            homo = dict({i: 0 for i in range(nmin, nmax)})
            comp = dict({i: 0 for i in range(nmin, nmax)})

            x, y = preprocess(df)

            for n in range(nmin, nmax):
                homo[n], comp[n] = Pitcher.get_h_and_c_scores(x, y, n)

            return Pitcher._estimate_n_clusters(homo, comp)

        except EOFError:
            raise EOFError('get_n_clusters(): DataFrame lacks sufficient quality data.')
