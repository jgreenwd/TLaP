from settings import STD_PITCHES_REMOVED, PITCH_KEYS
# -------------------------------------------------------- #
# ------ FUNCTIONS USED IN PROJECT JUPYTER NOTEBOOK ------ #
# -------------------------------------------------------- #


def _get_pitch_count(df, pid):
    """ Return number of pitches thrown in gameplay by pid

        :param df:  (DataFrame) valid pitch data
        :param pid: (Integer) valid pitcher_id

        :return:    (Integer) number of pitches thrown in gameplay by pid
    """
    if len(df) == 0:
        raise EOFError('_get_pitch_count: DataFrame is empty')

    return len(df[df.pitcher_id == pid])


def filter_wild_pitches(df, dist=5):
    """ Return pitches within 'dist' feet of the strike zone center

        :param df:   (DataFrame) valid pitch data
        :param dist: (Integer) limiting distance from strike zone center (Default 4)

        :return:     (DataFrame) filtered pitch data
    """
    if len(df) == 0:
        raise EOFError('_filter_wild_pitches: DataFrame is empty')

    return df[(df.px.abs() < dist) & (df.pz.abs() < dist)]


def encode_p_throws(df):
    """ Return DataFrame with p_throws categorical strings encoded as integers """
    if len(df) == 0:
        raise EOFError('_encode_p_throws: DataFrame is empty')

    return df.replace({'p_throws': {'R': 0, 'L': 1}})


def filter_pitches(df, remove=None):
    """ Return pitches whose pitch_type is not found in STD_PITCHES_REMOVED

        :param df:     (DataFrame) valid pitch data
        :param remove: (List) pitch types to remove from df
    """
    if len(df) == 0:
        raise EOFError('_filter_pitches: DataFrame is empty')

    _pitches_to_remove = set(STD_PITCHES_REMOVED).union(remove) if remove else set(STD_PITCHES_REMOVED)

    return df[~df.pitch_type.isin(list(_pitches_to_remove))]


def encode_pitch_type(df):
    """ Return DataFrame with pitch_type categorical strings encoded as integers

        :param df:   (DataFrame) valid pitch data
    """
    if len(df) == 0:
        raise EOFError('_encode_pitch_type: DataFrame is empty')

    return df.replace(PITCH_KEYS)


def separate_target_values(df):
    """ Return tuple of (modified DataFrame, extracted target values) """
    if len(df) == 0:
        raise EOFError('_separate_target_values(): DataFrame is empty')

    y = df.pitch_type.values
    x = df.drop('pitch_type', axis=1)

    return x, y


def preprocess(df, confidence=False, accuracy=False):
    """ Return (X, y) processed for training/testing

        :param df:         (DataFrame) valid pitch data
        :param confidence: (Boolean) remove pitches with low type_confidence?
        :param accuracy:   (Boolean) remove wild pitches?

        :return:           (Tuple: (X, y)) pitch data processed & encoded for training/testing
     """
    columns = df.columns

    for feature in ['pitcher_id', 'px', 'pz', 'p_throws', 'pitch_type']:
        if feature not in columns:
            raise AttributeError(f'preprocess: Invalid DataFrame - {feature} feature is not present.')

    try:
        _x = df.dropna()

        if confidence:
            _x = _x[_x.type_confidence >= 1]

        if accuracy:
            _x = filter_wild_pitches(_x)

        _x = encode_p_throws(_x)
        _x = filter_pitches(_x)
        _x = encode_pitch_type(_x)
        _x, _y = separate_target_values(_x)

        return _x, _y

    except EOFError:
        raise EOFError('preprocess(): DataFrame lacks sufficient quality data.')
