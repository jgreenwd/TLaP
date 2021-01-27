import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
import ipywidgets as widgets
import scipy.stats
from IPython import display
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score, adjusted_rand_score, homogeneity_score, completeness_score
from sklearn.preprocessing import robust_scale as scale
from Pitcher import Pitcher
from utils import preprocess, filter_pitches
from settings import PALETTE, WORKING_DIR, PITCH_KEYS, STD_PITCHES_REMOVED


class Dashboard:
    def __init__(self, df):
        # MODEL - State
        self.data = df
        self.pitcher = Pitcher(df.pitcher_id[0], list(np.unique(df.ab_id // 1000000)))
        self._selected_season = self.pitcher.get_seasons()[0]
        self._n_clusters = 4
        self._accuracy = False

        # VIEW - Components
        self._LOADING_GIF = None
        self._estimate = widgets.Output()
        self._output = widgets.Output()

        # CONTROLLER - Interactivity
        self._SEASON_SELECT = None
        self._CLUSTER_SELECT = None
        self._ACCURACY_TOGGLE = None
        self._RENDER_BUTTON = None

        # Assembled components
        self._dashboard = widgets.HBox(children=[self._build_control_pane(), self._build_output_pane()],
                                       layout={'width': '820px', 'height': '640px'})

    # MODEL: State alteration methods
    def _set_selected_season(self, season):
        self._selected_season = season

    def _set_n_clusters(self, n):
        self._n_clusters = n

    def _set_accuracy(self, value):
        self._accuracy = value

    # VIEW: Construct screen representation objects
    def _load_loading_gif(self):
        # create local loading_gif resource for output pane
        if self._LOADING_GIF is None:
            try:
                file = open(WORKING_DIR + f'./data/loading2.gif', 'rb')
                self._LOADING_GIF = widgets.Image(value=file.read(), format='gif')
                file.close()
            except IOError:
                print('Loading image not found')

    def _display_estimate(self, b=None):
        self._estimate.clear_output(wait=True)
        ab_id_start, ab_id_end = self._get_season_range(self._selected_season)

        if self._LOADING_GIF is None:
            self._load_loading_gif()

        with self._estimate:
            display.Image(self._LOADING_GIF, height=100)
            self._estimate.clear_output(wait=True)
            Dashboard.plot_n_clusters(self.data[self.data.ab_id.between(ab_id_start, ab_id_end)])

        return b

    def _display_gmm(self, b=None):
        self._output.clear_output(wait=True)
        ab_id_start, ab_id_end = self._get_season_range(self._selected_season)

        if self._LOADING_GIF is None:
            self._load_loading_gif()

        with self._output:
            display.Image(self._LOADING_GIF, height=100)
            self._output.clear_output(wait=True)
            Dashboard.plot_gmm(self.data[self.data.ab_id.between(ab_id_start, ab_id_end)],
                               n_clusters=self._n_clusters, accuracy=self._accuracy)

        return b

    # CONTROLLER: User/Data Interaction
    def _build_control_pane(self):
        # select season data to view
        seasons = sorted(list(self.pitcher.get_seasons()))
        season_select_label = widgets.Label(value='Season:')
        self._SEASON_SELECT = widgets.RadioButtons(options=seasons, layout={'width': '88px'})
        seasons_sub_display = widgets.VBox(children=[season_select_label, self._SEASON_SELECT],
                                           layout={'width': ' 150px', 'height': '140px', 'margin': '0 0 0 35px'})
        self._SEASON_SELECT.observe((lambda s: self._set_selected_season(s.new)), names='value')
        self._SEASON_SELECT.observe((lambda s: self._display_estimate(s.new)), names='value')

        # select number of clusters to calculate
        def label_update(change):
            cluster_select_label.value = f'{change.new}-Clusters'

        cluster_select_label = widgets.Label(value=f'{self._n_clusters}-Clusters', layout={'margin': '0 0 0 30px'})
        self._CLUSTER_SELECT = widgets.IntSlider(min=2, max=7, value=4, readout=False,
                                                 layout={'width': '85px', 'margin': '0 0 0 20px'})
        cluster_sub_display = widgets.VBox(children=[cluster_select_label, self._CLUSTER_SELECT],
                                           layout={'width': '190px', 'height': '75px', 'margin': '0 0 0 0'})
        self._CLUSTER_SELECT.observe((lambda s: self._set_n_clusters(s.new)), names='value')
        self._CLUSTER_SELECT.observe(label_update, names='value')

        # option to filter data based on accuracy
        self._ACCURACY_TOGGLE = widgets.Checkbox(value=False, description='Accuracy',
                                                 layout={'width': '190px', 'height': '75px', 'margin': '0 0 0 -65px'})
        self._ACCURACY_TOGGLE.observe((lambda s: self._set_accuracy(s.new)), names='value')

        # plot results
        self._RENDER_BUTTON = widgets.Button(description='Render', layout={'width': '100px', 'height': '28px',
                                                                           'margin': '1px 0 1px 10px'})
        self._RENDER_BUTTON.on_click(self._display_gmm)

        controls_layout = widgets.Layout(display='flex', flex_flow='column nowrap', align_content='center',
                                         align_items='flex-start', justify_content='flex-start', width='190px')

        return widgets.VBox(layout=controls_layout, children=[seasons_sub_display, cluster_sub_display,
                                                              self._ACCURACY_TOGGLE, self._RENDER_BUTTON])

    def _build_output_pane(self):
        return widgets.VBox(children=[self._estimate, self._output],
                            layout=widgets.Layout(display='flex', flex_flow='column nowrap', align_content='center',
                                                  align_items='center', justify_content='flex-start', width='505px',
                                                  height='640px'))

    def render_dashboard(self):
        self._display_estimate()
        self._load_loading_gif()
        return self._dashboard

    # Data Processing & Rendering
    @staticmethod
    def _get_season_range(value):
        begin = value * 1000000
        end = (value + 1) * 1000000

        return begin, end

    @staticmethod
    def plot_n_clusters(df, n_min=2, n_max=8):
        """ Render estimate of n_clusters using Gaussian Mixture Model

            :param df:    (DataFrame) valid pitch data
            :param n_min: (Integer) minimum number of clusters to test
            :param n_max: (Integer) maximum number of clusters to test
        """
        try:
            if len(df) > 250000:
                raise OverflowError('plot_n_clusters(): DataFrame is too large to process. Limit 250,000 entries')

            _x, _y = preprocess(df, accuracy=True)
            _x = scale(_x)

            if len(_x) < 1000:
                raise EOFError('plot_n_clusters(): DataFrame contains insufficient number of quality data points')

            # clustering metrics
            _homogeneity = dict({i: 0 for i in range(n_min, n_max)})
            _completeness = dict({i: 0 for i in range(n_min, n_max)})

            for n in range(n_min, n_max):
                _homogeneity[n], _completeness[n] = Pitcher.get_h_and_c_scores(_x, _y, n)

            # ----- PLOT METRICS -----
            fig, ax = plt.subplots(figsize=(10, 2))
            fig.suptitle(f'Gaussian Mixture Model    Sample size: {len(_x)}', fontsize=15)

            ax.set_ylim((0, 1))
            ax.set_xlabel('Number of Pitch Types', fontsize=13)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.plot(list(_homogeneity.keys()), list(_homogeneity.values()))
            ax.plot(list(_completeness.keys()), list(_completeness.values()))
            leg1 = ax.legend(['Homogeneity', 'Completeness'], loc='lower left', fontsize=12, frameon=False)
            plt.gca().add_artist(leg1)
            vert1 = plt.axvline(len(np.unique(_y)), color='#005D00', linestyle=(0, (1, 1.5)), zorder=5,
                                linewidth=5, label='n-Clusters: MLB-AM')
            ax.legend(handles=[vert1], loc='lower right', fontsize=12, frameon=False)

            plt.show()

        except EOFError as err:
            print(err)
        except OverflowError as err:
            print(err)

    @staticmethod
    def _sync_pitch_colors(dictionary):
        """ Return a List of Integers representing color indices in the Palette

            :param dictionary: (Dict) pitch counts
            """
        return [PITCH_KEYS[k] for k in dictionary]

    @staticmethod
    def _draw_ellipse(position, covariance, ax, **kwargs):
        """ Draw an ellipse with a given position and covariance """
        # SOURCE: https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            u, s, vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(u[1, 0], u[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

    @staticmethod
    def _get_cluster_centers(model, arr):
        """ Find & return cluster centers

            :param model: (GMM model instance)
            :param arr:   (ndarray) preprocessed, filtered, & scaled data

            :return: (ndarray) cluster centers
        """
        centers = np.empty(shape=(model.n_components, arr.shape[1]))
        for i in range(model.n_components):
            density = scipy.stats.multivariate_normal(cov=model.covariances_[i], mean=model.means_[i]).logpdf(arr)
            centers[i, :] = arr[np.argmax(density)]

        return centers

    @staticmethod
    def _build_gmm_plot(x, y, model, scores, counts, opacity):
        """ Render 2d plot of DataFrame using Gaussian Mixture Model with metrics and pitch counts

            :param x:       (Ndarray)
            :param y:       (Ndarray)
            :param model:   (GMM model reference)
            :param scores:  (Dictionary) of {Metric: Result} clustering metrics
            :param counts:  (Dictionary) of {Name: Count} pitch counts
            :param opacity: (Float)
        """
        import warnings

        # -- current known bug in matplotlib generates warning while attempting to set_yticklabels below
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # ----- PLOT FIGURE -----
            fig = plt.figure(figsize=(8, 8), facecolor='black')

            _centers = Dashboard._get_cluster_centers(model, x)
            _scaled_dot_size = (3 * model.predict_proba(x).max(1) ** 3)

            # -- Cluster display
            ax0 = fig.add_subplot(111, facecolor='black')
            ax0.scatter(x[:, 0], x[:, 1], c=y, s=_scaled_dot_size, cmap=ListedColormap(PALETTE, N=(int(np.max(y) + 1))))
            ax0.scatter(_centers[:, 0], _centers[:, 1], s=100, marker='o', color=(1, 0.5, 0.8, 1),
                        zorder=3, label='Cluster centers')
            legend = ax0.legend(bbox_to_anchor=(0, 0.01, 0.2, 0.05), framealpha=0, prop={'size': 14, 'weight': 600})
            ax0.set_ylim((ax0.get_ylim()[0], (np.max(_centers[:, 1]) * 2)))
            for text in legend.get_texts():
                plt.setp(text, color='snow')

            # -- add Decision Boundaries
            w_factor = opacity / model.weights_.max()
            for pos, covar, w in zip(model.means_, model.covariances_, model.weights_):
                Dashboard._draw_ellipse(pos, covar, ax0, alpha=w * w_factor, color='snow', zorder=2)

            # -- Metrics overlay
            ax1 = fig.add_subplot(331, facecolor=(0, 0, 0, 0))
            ax1.barh(list(scores.keys()), scores.values(), color=(0.6, 0.6, 0.6, 0.5),
                     alpha=0.55, edgecolor=(0, 0, 0, 0))
            ax1.tick_params(axis='x', labelsize=0)
            ax1.set_yticklabels([f'{v:0.2f} - {k}' for k, v in scores.items()], x=0.07,
                                c='snow', weight=600, ha='left', fontsize=14)
            ax1.set_position([0, 0.65, 0.98, 0.25])

            # -- Pitch counts overlay
            ax2 = fig.add_subplot(339, facecolor=(0, 0, 0, 0))
            ax2.barh(list(counts.keys()), [-1 * x for x in counts.values()], edgecolor=(0, 0, 0, 0),
                     color=[PALETTE[c] for c in Dashboard._sync_pitch_colors(counts)], alpha=0.65)
            ax2.set_yticks(np.arange(len(counts)))
            ax2.set_yticklabels(counts.keys(), c='snow', ha='center', x=0.975, weight=600, fontsize=14)
            ax2.tick_params(axis='x', labelsize=0)
            ax2.set_position([0.5, 0.1, 0.5, 0.25])
            for spine in ['top', 'right', 'bottom', 'left']:
                ax1.spines[spine].set_visible(False)
                ax2.spines[spine].set_visible(False)

            plt.show()

    @staticmethod
    def plot_gmm(df, n_clusters=None, accuracy=False, xaxis='pfx_x', yaxis='vy0', opacity=0.25):
        """ Render DataFrame using Gaussian Mixture Model

            :param df:         (DataFrame) pitch data
            :param n_clusters: (Integer) number of clusters to plot (Default None)
            :param accuracy:   (Boolean) filter dataframe based on accuracy (Default False)
            :param xaxis:      (String) column label in df to use for x-axis in plot (Default pfx_x)
            :param yaxis:      (String) column label in df to use for y-axis in plot (Default vy0)
            :param opacity:    (Float) decision boundary opacity
        """
        if xaxis not in df.columns:
            raise AttributeError(f'plot_gmm: {xaxis} not found in DataFrame')
        if yaxis not in df.columns:
            raise AttributeError(f'plot_gmm: {yaxis} not found in DataFrame')

        try:
            # Record & sort pitch counts alphabetically
            _pitch_counts = filter_pitches(df).pitch_type.value_counts().to_dict()
            _pitch_counts = {k: v for k, v in _pitch_counts.items() if k not in STD_PITCHES_REMOVED}

            _x, _y = preprocess(df, confidence=False, accuracy=accuracy)

            # Remove all unused features
            _x = _x.filter([xaxis, yaxis])
            _x = scale(_x)

            n_clusters = n_clusters if n_clusters else len(np.unique(_y))

            # Instantiate and score model
            _model = GaussianMixture(n_components=n_clusters, covariance_type='full')
            _preds = _model.fit_predict(_x, _y)
            _scores = {'V-measure': v_measure_score(_y, _preds),
                       'Completeness': completeness_score(_y, _preds),
                       'Homogeneity': homogeneity_score(_y, _preds),
                       'ARI': adjusted_rand_score(_y, _preds)}

            Dashboard._build_gmm_plot(_x, _y, _model, _scores, _pitch_counts, opacity)

        except EOFError:
            raise EOFError('plot_gmm: DataFrame lacks sufficient quality data.')
