"""Build features from raw data
"""
import os.path as op
import numpy as np
from sklearn import pipeline
from sklearn import preprocessing


np.set_printoptions(precision=4, suppress=True)

feature_types = {'continuous': ['resting_blood_pressure', 'serum_cholesterol_mg_per_dl',
                                'oldpeak_eq_st_depression', 'age', 'max_heart_rate_achieved'],
                 'ordinal': ['slope_of_peak_exercise_st_segment', 'num_major_vessels',
                             'resting_ekg_results'],
                 'categorical': ['thal', 'chest_pain_type', 'fasting_blood_sugar_gt_120_mg_per_dl',
                                 'sex', 'exercise_induced_angina']}


def build_preprocessors(data):
    """Build a preprocessor for each variable.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame

    Returns
    -------
    A dict with keys as the feature names and values as sklearn.pipeline.Pipeline to preprocess
    each feature
    """
    preprocessing_steps = {}
    for column in data.columns:
        values = data[column].values.reshape(-1, 1)
        if column in feature_types['continuous']:
            values = values.astype('float32')
            steps = pipeline.Pipeline([('standard_scaler', preprocessing.StandardScaler())])

        elif column in feature_types['ordinal']:
            values = values.astype('float32')
            steps = pipeline.Pipeline([('standard_scaler', preprocessing.StandardScaler())])

        elif column in feature_types['categorical']:
            one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')
            steps = pipeline.Pipeline([('one_hot_encoder', one_hot_encoder)])

        _ = steps.fit(values)
        preprocessing_steps[column] = steps

    return preprocessing_steps


def preprocess_data(data, preprocessors):
    """Preprocess data given a set of preprocessing steps for each variable

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
    preprocessors : dict of sklearn.pipeline.Pipeline

    Returns
    -------
    A numpy array of size (number of rows, number of features) containing the preprocesse
    data
    """

    features = []
    for column in data:
        feat = preprocessors[column].transform(data[column].values.reshape(-1, 1))
        features.append(feat)

    return features


class ConcatenateFeatures():
    """Translate between individual and concatenated feature spaces
    """
    def fit(self, features, names=None):
        """Build the concatenator from a list of feature spaces. A "feature space"
        is a conceptually grouped set of variables

        Parameters
        ----------
        features : list of np.ndarray
        names : list of str, optional
            If provided, the name of each feature space
        """
        if names is None:
            names = [f"feature{name}"
                     for name in range(len(features))]

        feature_sizes = []
        feature_names = []
        feature_splits = [0]
        for feature, name in zip(features, names):
            size = feature.shape[1]
            feature_sizes.append(size)
            feature_splits.append(feature_splits[-1] + size)
            feat_names = [f"{name}_{index}" for index in range(size)]
            feature_names.extend(feat_names)

        feature_splits = feature_splits[1:-1]

        self._feature_sizes = feature_sizes
        self._feature_splits = feature_splits
        self._feature_names = feature_names

        return self

    def transform(self, features):
        """Concatenate a list of feature spaces into a single feature array
        """
        return np.hstack(features)

    def inverse_transform(self, features):
        """Split a feature array into a list of individual feature spaces
        """
        return np.array_split(features, self._feature_splits, axis=1)
