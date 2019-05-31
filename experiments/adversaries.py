import json
import pickle
import re
import sys
import os
import datetime
from subprocess import run as run_cmd, PIPE

import click
import pandas as pd
from tqdm import tqdm

wd = os.getcwd() + '/'
sys.path.append(wd + '../')
# Anchors
sys.path.append(wd + 'anchor/')
# LORE
sys.path.append(wd + 'lorem/')
# BRL
sys.path.append(wd + 'sklearn-expertsys/')

from anchor.anchor_tabular import AnchorTabularExplainer
from lorem.datamanager import prepare_dataset
from lorem.lorem import LOREM
from trepan.transform import trepan_to_rules
from trepan.trepan import TREPAN
import pysbrl

from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

from numpy import array, argwhere, hstack, float64, int64, inf, rint, argmax

from discretize import EntropyDiscretizer
from rule_based_classifier import RuleBasedClassifier
# from RuleListClassifier import RuleListClassifier

from models import Rule

dt_hyperparameters = {
    'max_depth': (2, 4, 8, 16, 32),
    'min_samples_split': (.1, .2, .5, .75),
    'min_samples_leaf': (1, 2, 5, 10, 25)
}


def __encode_for_jrbc(df):
    """Encode @df for JRBC algoriths.

    Arguments:
        df (DataFrame): DataFrame to encode, including the labels.

    Returns:
        (tuple): Tuple (DataFrame, list, EntropyDiscretizer)
                holding the encoded dataset, a list of pairs
                (feature, label_encoder) and the discretizer.
    """
    encoded_df = df.copy()
    data = encoded_df.values[:, :-1]
    labels = encoded_df.values[:, -1]
    types = encoded_df.dtypes

    non_discretized_features = argwhere(types != object).flatten().tolist()
    discretized_features = argwhere(types == object).flatten().tolist()
    feature_names = encoded_df.columns

    discretizer = None
    discretized_data = data
    if len(non_discretized_features) > 0:
        discretizer = EntropyDiscretizer(data, discretized_features, feature_names, labels=labels.astype(int))
        discretized_data = discretizer.discretize(data)

    encoded_df = DataFrame(hstack((discretized_data, labels.reshape(-1, 1))), columns=encoded_df.columns)
    scaling_factors = [0]
    label_encoders = {}
    for feature in encoded_df.columns:
        label_encoder = LabelEncoder().fit(encoded_df[feature])
        encoded_df[feature] = label_encoder.transform(encoded_df[feature])
        encoded_df[feature] += sum(scaling_factors)
        scaling_factors.append(encoded_df[feature].unique().shape[0])
        label_encoders[feature] = label_encoder

    return encoded_df, label_encoders, discretizer, scaling_factors


def _decode_from_jrbc(df, encoders, discretizer, scaling_factors):
    """Scale @x, @y from CPAR's encoding.
    Each feature is scaled from positive integers and with values larger then the preceding.
    Returns the data scaled back to its original encoding.
    """
    df_ = df.copy()

    # Scale in order to have sorted rows
    for (i, encoder), scaling_factor in zip(encoders, scaling_factors):
        df_[i] = encoder.inverse_transform(df_[i] - scaling_factor)
    df_ = discretizer.undiscretize(df_)

    return df_


def _binary_discretization(df, buckets=4, name=None):
    """Discretize the given `x` into binary `buckets` per feature.
    Args:
        df (DataFrame): The data.
        buckets (int): Number of buckets.
        name (str): If not None, dump discretization info on `name`.json
    Returns:
        (DataFrame): Binarly discretized DataFrame.
    """
    m, n = df.shape
    columns = df.columns
    bucket_size = m // buckets
    x = df.values
    binned_x = df.copy().values[:, :-1]

    continuous_features = [i for i in range(n - 1) if len(set(x[:, i])) > 2]
    discrete_features = [i for i in range(n - 1) if len(set(x[:, i])) <= 2]
    buckets_argindices = [(bucket_size * i, bucket_size * (i + 1)) for i in range(buckets)]
    buckets_argindices[-1] = (buckets_argindices[-1][0], buckets_argindices[-1][1] - 1)
    full_thresholds = {}

    for feature in continuous_features:
        values = sorted(set(x[:, feature]))
        bucket_size = len(values) // buckets
        thresholds = [(int(values[i * bucket_size]), int(values[(i + 1) * bucket_size])) for i in range(buckets - 1)]
        full_thresholds[columns[feature]] = thresholds

        for bucket_id, (lower_bound, upper_bound) in zip(range(buckets), thresholds):
            binned_x[:, feature][(lower_bound <= binned_x[:, feature]) &
                                 (binned_x[:, feature] < upper_bound)] = bucket_id
        full_thresholds[columns[feature]].append((full_thresholds[columns[feature]][-1][1],
                                                  int(binned_x[:, feature].max())))
        binned_x[:, feature][binned_x[:, feature] > buckets] = buckets - 1

    binned_df = pd.DataFrame(binned_x, columns=columns[:-1])
    continuous_columns = [columns[i] for i in continuous_features]
    discrete_columns = [columns[i] for i in discrete_features]

    for feature in continuous_columns:
        binned_df[feature] = binned_df[feature].astype(int).astype(str)

    if n - 1 != len(discrete_features):
        binarized_x = pd.concat([pd.get_dummies(binned_df[continuous_columns], prefix_sep='='), binned_df[discrete_columns]],
                                axis='columns')
        binarized_x['y'] = df.values[:, -1]
    else:
        binarized_x = pd.concat([pd.get_dummies(binned_df[continuous_columns]), df['y']], axis='columns')

    # Columns
    full_cols = []
    cols = binarized_x.columns.tolist()
    for col in cols:
        if '_' in col and col[:col.index('_')] in full_thresholds:
            feature, interval = int(col.split('='))
            interval = full_thresholds[feature][interval]
            full_cols.append(feature + '=' + str(interval))
        else:
            full_cols.append(col)
    binarized_x.columns = full_cols

    if name is not None:
        with open(name + '.discretization.json', 'w') as log:
            json.dump(full_thresholds, log)

    return binarized_x


def _label_discretization(df, buckets=4, max_elements_per_feature=10):
    """Discretize the given `x` into integer `buckets`.
    Args:
        df (DataFrame): The data.
        buckets (int): Number of buckets.
        max_elements_per_feature (int): All features with less than
                                        `max_elements_per_feature` are
                                        not discretized.
    Returns:
        (x): Labelized `x`.
    """
    x = df.values
    m, n = x.shape
    bucket_size = m // buckets
    binned_x = x.copy()

    continuous_features_indices = [c for c in range(n - 1) if df.dtypes[c] != object and
                                   len(df[df.columns[c]].unique()) > buckets]
    continuous_features = [df.columns[i] for i in continuous_features_indices]
    categorical_features = [df.columns.tolist()[c] for c in range(n - 1) if df.dtypes[c] == object]
    buckets_argindices = [(bucket_size * i, bucket_size * (i + 1)) for i in range(buckets)]
    buckets_argindices[-1] = (buckets_argindices[-1][0], buckets_argindices[-1][1] - 1)
    full_thresholds = {}

    for feature in continuous_features_indices:
        values = sorted(set(x[:, feature]))
        bucket_size = len(values) // buckets
        thresholds = [(values[i * bucket_size], values[(i + 1) * bucket_size]) for i in range(buckets - 1)]
        full_thresholds[feature] = thresholds

        for bucket_id, (lower_bound, upper_bound) in zip(range(buckets), thresholds):
            binned_x[:, feature][(lower_bound <= binned_x[:, feature].astype(float64)) &
                                 (binned_x[:, feature].astype(float64) < upper_bound)] = str(bucket_id)
        binned_x[:, feature][binned_x[:, feature].astype(float64) > bucket_size] = buckets - 1

    if len(categorical_features) > 0:
        labelized_x = pd.concat([pd.DataFrame(binned_x[:, continuous_features_indices], columns=continuous_features),
                                 pd.get_dummies(df[categorical_features], prefix_sep='='),
                                 df.iloc[:, -1]],
                                axis='columns')
    else:
        labelized_x = pd.concat([pd.DataFrame(binned_x[:, continuous_features_indices], columns=continuous_features),
                                 df.iloc[:, -1]],
                                axis='columns')

    return labelized_x


def dataset_to_corels_file(df, output):
    """Generate the binary output suitable for a CORELS run from
    the given `df` in file `output`.
    Args:
        df (DataFrame): The data to binarize.
        output (str): The output file prefix. The output files
                        are `output`.data.corels and `output`.label.corels
    """
    binaryzed_df = _binary_discretization(df, name=output)
    columns = binaryzed_df.columns
    descriptions = columns.tolist()[:-1]
    descriptions = ['{' + desc.replace('=', ':') + '} ' for desc in descriptions]
    values = [binaryzed_df[c].astype(int).astype(str) for c in columns[:-1]]
    feature_strings = [' '.join(feature_values) + '\n' for feature_values in values]
    feature_strings = [description + value for description, value in zip(descriptions, feature_strings)]
    feature_strings = ''.join(feature_strings)
    y = binaryzed_df.astype(int).values[:, -1]
    not_y = ((y + 1) % 2).astype(int)
    class_strings = ['{y:1} ' + ' '.join(y.astype(str)),
                     '{y:0} ' + ' '.join(not_y.astype(str))]
    class_strings = '\n'.join(class_strings)

    # Dump on disk
    with open(output + '.data.corels', 'w') as log:
        log.write(feature_strings)
    with open(output + '.label.corels', 'w') as log:
        log.write(class_strings)

    return feature_strings, class_strings


def dataset_to_sbrl_file(df, output):
    """Generate the binary output suitable for a SBRL run from
    the given `df` in file `output`.
    Args:
        df (DataFrame): The data to binarize.
        output (str): The output file prefix. The output files
                        are `output`.data.sbrl and `output`.label.sbrl
    """
    binaryzed_df = _binary_discretization(df, name=output)
    columns = binaryzed_df.columns
    descriptions = columns.tolist()[:-1]
    descriptions = ['{' + desc.replace('=', ':') + '} ' for desc in descriptions]
    values = [binaryzed_df[c].astype(int).astype(str) for c in columns[:-1]]
    feature_strings = [' '.join(feature_values) + '\n' for feature_values in values]
    feature_strings = [description + value for description, value in zip(descriptions, feature_strings)]
    feature_strings = ''.join(feature_strings)
    y = binaryzed_df.astype(int).values[:, -1]
    not_y = ((y + 1) % 2).astype(int)
    class_strings = ['{y:1} ' + ' '.join(y.astype(str)),
                     '{y:0} ' + ' '.join(not_y.astype(str))]
    class_strings = '\n'.join(class_strings)
    header = 'n_items: ' + str(df.shape[1]) + '\n'
    header = header + 'n_samples: ' + str(df.shape[0]) + '\n'
    header_labels = 'n_items: 2\n'
    header_labels = header_labels + 'n_samples: ' + str(df.shape[0]) + '\n'

    # Dump on disk
    with open(output + '.data.sbrl', 'w') as log:
        log.write(header + feature_strings)
    with open(output + '.label.sbrl', 'w') as log:
        log.write(header_labels + class_strings)


def _decision_set_discretization(df):
    """Discretize the given dataset `df` to accomodate the
    Decision Set internal representation.
    Args:
        df (DataFrame): The dataset.
    Returns:
        (DataFrame): The discretized dataset.
    """
    ids_df = pd.DataFrame(_label_discretization(df))

    return ids_df


def _decision_set_rule(decision_set_rule):
    """Encode the given Decision Rule into a `Rule`.
    Args:
        decision_set_rule (rule): Rule in the Decision Set format.
    Returns:
        (Rule): Rule in the `Rule` format.
    """
    return None


def _jrbc(alg, df):
    """
    Build and train a Java Rule-Based Classifier adversary.

    Arguments:
        df (ndarray): The TR set.

    Returns:
        (RuleBasedClassifier): A Java Rule-Based Classifier trained on @alg algorithm.
    """
    jrbc = RuleBasedClassifier(alg, options='-Xmx4G')
    df_, encoders, discretizer, scaling_factors = __encode_for_jrbc(df)
    x, y = df_.values[:, :-1], df_.values[:, -1]
    jrbc.fit(x, y, verbose=False)

    return jrbc, encoders, discretizer, scaling_factors


def _cpar(df, path, cwd):
    """Create a CPAR model from @patterns.
    Arguments:
        df (DataFrame): A numeric-only ndarray holding the TR data.

    Returns:
        (CPAR): A CPAR classifier.
    """
    os.chdir(path)
    cpar, encoders, discretizer, scaling_factors = _jrbc('CPAR', df)
    cpar = CPAR(cpar, encoders, discretizer, scaling_factors, df.dtypes, path, cwd)
    os.chdir(cwd)

    return cpar


def _foil(df, path, cwd):
    """Create a CPAR model from @patterns.
    Arguments:
        patterns (ndarray): A numeric-only ndarray holding the TR data.

    Returns:
        (FOIL): A FOIL classifier.
    """
    os.chdir(path)
    foil, encoders, discretizer, scaling_factors = _jrbc('FOIL', df)
    foil = FOIL(foil, encoders, discretizer, scaling_factors, df.dtypes, path, cwd)
    os.chdir(cwd)

    return foil


def _rules_from_cpar(cpar):
    """Extract the rules generated by @cpar.

    Arguments:
        cpar (CPAR): A CPAR classifier.

    Returns:
        (set): A set of rules extracted from @cpar.
    """
    cpar_rules = cpar.cpar.rules
    encoders = cpar.encoders
    scaling_factors = cpar.scaling_factors
    if len(cpar_rules) == 0:
        return []

    outcomes = array(list(map(lambda r: r[0], cpar_rules))) + 2
    outcomes -= sum(scaling_factors)
    ranges = list(map(lambda r: r[1], cpar_rules))
    ranges = list(filter(lambda r: len(r) > 0, ranges))
    decoded_ranges = []
    for rule_ranges in ranges:
        ranges_dic = {}
        for feature in rule_ranges.keys():
            # Continuous
            if cpar.types[feature] == float64:
                bin = rule_ranges[feature] - sum(scaling_factors[:feature + 1])
                interval = cpar.discretizer.names[feature][bin]
                splits = interval.split('<')
                # feature in [a, b]
                if len(splits) == 3:
                    ranges_dic[feature] = (float64(splits[0]), float64(splits[2][2:]))  # Skip first character, is a '= '
                # feature < a,
                elif len(splits) == 2:
                    ranges_dic[feature] = (-inf, float64(splits[1][1:]))  # Skip first character, is an '='
                # feature > b
                else:
                    val = float64(interval.split('> ')[1])
                    ranges_dic[feature] = (val, +inf)
            # Integers
            elif cpar.types[feature] == int64:
                bin = rule_ranges[feature] - sum(scaling_factors[:feature + 1])
                interval = cpar.discretizer.names[feature][bin]
                splits = interval.split('<')
                # feature in [a, b]
                if len(splits) == 3:
                    ranges_dic[feature] = (int(rint(float64(splits[0]))),
                                           int(rint(float64(splits[2][2:]))))  # Skip first two characters, is a '= '
                # feature < a,
                elif len(splits) == 2:
                    ranges_dic[feature] = (-inf, int(rint(float64(splits[1][1:]))))  # Skip first character, is an '='
                # feature > b
                else:
                    val = rint(int(float64(interval.split('> ')[1])))
                    ranges_dic[feature] = (val, +inf)
            # Categorical
            else:
                val = encoders[feature].inverse_transform(rule_ranges[feature] - sum(scaling_factors[:feature + 1]))
                ranges_dic[feature] = (val, val)

        decoded_ranges.append(ranges_dic)

    cpar_rules = [Rule(premises={}, consequence=outcome) for outcome in outcomes]
    for rule, rule_range in zip(cpar_rules, decoded_ranges):
        rule.premises = rule_range

    return cpar_rules


def _rules_from_foil(foil):
    """Extract the rules generated by @cpar.

    Arguments:
        foil (FOIL): A FOIL classifier.
        encoders (list): Encoders to go from numerical to original
                            encoding. If None, the numerical encoding
                            is returned.
    Returns:
        {set): A set of rules extracted from @cpar.
    """
    encoders = foil.encoders
    scaling_factors = foil.scaling_factors
    foil_rules = foil.foil.rules
    if len(foil_rules) == 0:
        return []

    outcomes = array(list(map(lambda r: r[0], foil_rules))) + 2
    outcomes -= sum(scaling_factors)
    ranges = list(map(lambda r: r[1], foil_rules))
    ranges = list(filter(lambda r: len(r) > 0, ranges))
    decoded_ranges = []
    for rule_ranges in ranges:
        ranges_dic = {}
        for feature in rule_ranges.keys():
            # Continuous
            if foil.types[feature] == float64:
                bin = rule_ranges[feature] - sum(scaling_factors[:feature + 1])
                interval = foil.discretizer.names[feature][bin]
                splits = interval.split('<')
                # feature in [a, b]
                if len(splits) == 3:
                    ranges_dic[feature] = (float64(splits[0]), float64(splits[2][2:]))  # Skip first character, is a '= '
                # feature < a,
                elif len(splits) == 2:
                    ranges_dic[feature] = (-inf, float64(splits[1][1:]))  # Skip first character, is an '='
                # feature > b
                else:
                    val = float64(interval.split('> ')[1])
                    ranges_dic[feature] = (val, +inf)
            # Integers
            elif foil.types[feature] == int64:
                bin = rule_ranges[feature] - sum(scaling_factors[:feature + 1])
                interval = foil.discretizer.names[feature][bin]
                splits = interval.split('<')
                # feature in [a, b]
                if len(splits) == 3:
                    ranges_dic[feature] = (int(rint(float64(splits[0]))),
                                           int(rint(float64(splits[2][2:]))))  # Skip first two characters, is a '= '
                # feature < a,
                elif len(splits) == 2:
                    ranges_dic[feature] = (-inf, int(rint(float64(splits[1][1:]))))  # Skip first character, is an '='
                # feature > b
                else:
                    val = rint(int(float64(interval.split('> ')[1])))
                    ranges_dic[feature] = (val, +inf)
            # Categorical
            else:
                val = encoders[feature].inverse_transform(rule_ranges[feature] - sum(scaling_factors[:feature + 1]))
                ranges_dic[feature] = (val, val)

        decoded_ranges.append(ranges_dic)

    foil_rules = [Rule(premises={}, consequence=outcome) for outcome in outcomes]
    for rule, rule_range in zip(foil_rules, decoded_ranges):
        rule.premises = rule_range

    return foil_rules


def _rule_from_anchor(anchor):
    """Extract a rule from an ANCHOR.
    Args:
        (AnchorExplanation): The ANCHOR explanation.
    Returns:
        (Rule): An explanation.
    """
    tokens = list(map(lambda name: name.split(' '), anchor.names()))
    premises = {}

    for token in tokens:
        if len(token) == 3:
            if token[1] == '<=' or token[1] == '<':
                premises[int(token[0])] = (-inf, float(token[2]))
            elif token[1] == '>=' or token[1] == '>':
                premises[int(token[0])] = (float(token[2]), +inf)
        else:
            premises[int(token[2])] = (float(token[0]), float(token[4]))

    rule = Rule(premises=premises, consequence=int(anchor.exp_map['prediction']))

    return rule


def _rule_from_lore(explanation):
    """Extract a rule from an ANCHOR.
    Args:
        (Explanation): The LORE explanation.
    Returns:
        (Rule): An explanation.
    """
    consequence = int(explanation.rule.cons)
    premises = explanation.rule.premises
    features = [int(condition.att) for condition in premises]
    ops = [premise.op for premise in premises]
    values = [float(premise.thr) for premise in premises]
    values_per_feature = {feature: [val for f, val in zip(features, values) if int(f) == feature]
                          for feature in features}
    ops_per_feature = {feature: [op for f, op in zip(features, ops) if f == feature]
                       for feature in features}
    output_premises = {}
    for f in features:
        values, operators = values_per_feature[f], ops_per_feature[f]
        # 1 value, either <= or >
        if len(values) == 1:
            if operators[0] == '<=':
                output_premises[f] = (-inf, values[0])
            else:
                output_premises[f] = (values[0], +inf)
        # 2 values, < x <=
        else:
            output_premises[f] = (min(values), max(values))

    rule = Rule(premises=output_premises, consequence=consequence)

    return rule


def _rule_from_trepan(trepan):
    """Convert the provided Trepan `rule` to a `Rule`.
    Args:
        trepan (TREPAN): A Trepan instance.
    Returns:
        (list): A list of `Rule` object.
    """
    rules = trepan_to_rules(trepan)
    rules = [Rule(rule.premises, rule.consequence) for rule in rules]

    return rules


def __children(v, edges):
    return list(map(lambda edge: int(edge[1][1:]),
                    list(filter(lambda edge: edge[0] == v, list(edges)))))


def __all_paths(tree):
    """
    Retrieve all the possible paths in @tree.

    Arguments:
        tree {): The decision tree internals.

    Returns:
        (list): A list of list of indices:[path_1, path_2, .., path_m]
                    where path_i = [node_1, node_l].
    """
    paths = [[0]]
    l_child = tree.children_left[0]
    r_child = tree.children_right[0]

    if tree.capacity == 1:
        return paths

    paths = paths + \
            __rec_all_paths(tree, r_child, [0], +1) + \
            __rec_all_paths(tree, l_child, [0], -1)
    paths = sorted(set(map(tuple, paths)), key=lambda p: len(p))

    return paths


def __rec_all_paths_yadt(tree, node, current_path):
    """
    Recursive call for the @all_paths function.

    Arguments:
        tree (): The decision tree internals.
        node (int): The node whose path to expand.
        current_path (list): The current path root-> @node.

    Returns:
        (list): The enriched path.
    """
    if tree.node['n' + str(node)]['label'].count('/') > 0:
        return current_path + [node]
    else:
        node_children = __children('n' + str(node), list(tree.edges))

        res = []
        for child in node_children:
            children_paths = __rec_all_paths_yadt(tree, child, current_path + [node])
            res.append(children_paths)

        return res


def __rec_all_paths(tree, node, path, direction):
    """
    Recursive call for the @all_paths function.

    Arguments:
        tree (): The decision tree internals.
        node (int): The node whose path to expand.
        path (list): The path root-> @node.
        direction (int):  +1 for right child, -1 for left child.
                            Used to store the actual traversal.

    Returns:
        (list): The enriched path.
    """
    # Leaf
    if tree.children_left[node] == tree.children_right[node]:
        return [path + [node * direction]]
    else:
        path_ = [path + [node * direction]]
        l_child = tree.children_left[node]
        r_child = tree.children_right[node]

        return path_ + \
               __rec_all_paths(tree, r_child, path_[0], +1) + \
               __rec_all_paths(tree, l_child, path_[0], -1)


def _rules_from_dt(decision_tree):
    """Extract the rules   applied by @dt for the provided
    @patterns, i.e. the path followed by each path when
    classified by @dt.

    Arguments:
        decision_tree (DecisionTreeClassifier): The decision tree whose rules to extract.

    Returns:
        (set): A set of rules extracted from `decision_tree`.
    """
    tree = decision_tree.tree_

    paths = __all_paths(tree)
    paths = list(filter(lambda path: len(path) > 1, paths))
    consequences = [argmax(tree.value[abs(path[-1])]) for path in paths]
    features = [list(map(lambda node: tree.feature[abs(node)], path[:-1])) for path in paths]
    thresholds = [list(map(lambda node: tree.threshold[abs(node)], path[:-1])) for path in paths]
    rules = [Rule.fromarrays(feature_list, thresholds_list, consequence_list, paths_list)
             for feature_list, thresholds_list, consequence_list, paths_list
             in zip(features, thresholds, consequences, paths)]

    return rules


def _rules_from_decision_set(decision_set):
    """
    Extract rules from the given `decision_set`.
    Args:
        decision_set (DecisionSet): The Decision Set instance.
    Returns:
        (list): List of `Rule`.
    """
    rules = decision_set.rules
    # TODO: Implement
    raise ValueError('To implement')

    return rules


class CPAR:
    """CPAR classifier."""

    def __init__(self, cpar, encoders, discretizer, scaling_factors, types, path, cwd):
        """Constructor.
        Arguments:
            cpar: CPAR instance.
            encoders (list): Encoders.
            discretizer (BaseDiscretizer): Discretizer to scale data for CPAR.
            scaling_factors (iterable): Scaling factors to scale data for CPAR.
            types (iterable): Data types.
            path (str): Working directory.
            cwd (str): Current working directory.
        """
        self.cpar = cpar
        self.encoders = encoders
        self.discretizer = discretizer
        self.scaling_factors = scaling_factors
        self.types = types
        self.path = path
        self.cwd = cwd

    def predict(self, x):
        """Predict x.
        Arguments:
            x (ndarray): Data to predict.
        Returns:
            (ndarray): Predictions on x.
        """
        x_ = x.copy()
        os.chdir(self.path)
        x_ = _decode_from_jrbc(DataFrame(x_), self.encoders, self.discretizer, self.scaling_factors)
        y = self.cpar.predict(x_)
        scaled_y = y - self.scale[-1]
        os.chdir(self.cwd)

        # Fix for missing values in the TR
        scaled_y -= min(scaled_y)

        return scaled_y


class FOIL:
    """FOIL classifier."""

    def __init__(self, foil, encoders, discretizer, scaling_factors, types, path, cwd):
        """Constructor.
        Arguments:
            cpar: FOIL instance.
            encoders (list): Encoders.
            discretizer (BaseDiscretizer): Discretizer to scale data for FOIL.
            scaling_factors (iterable): Scaling factors to scale data for FOIL.
            types (iterable): Data types.
            path (str): Working directory.
            cwd (str): Current working directory.
        """
        self.foil = foil
        self.encoders = encoders
        self.discretizer = discretizer
        self.scaling_factors = scaling_factors
        self.types = types
        self.path = path
        self.cwd = cwd

    def predict(self, x):
        """Predict x.
        Arguments:
            x (ndarray): Data to predict.
        Returns:
            (ndarray): Predictions on x.
        """
        x_ = x.copy()
        os.chdir(self.path)

        for i, scale in enumerate(self.scale[:-1]):
            x_[:, i] += scale

        y = self.foil.predict(x_)
        scaled_y = y - self.scale[-1]
        os.chdir(self.cwd)

        # Fix for missing values in the TR
        scaled_y -= min(scaled_y)

        return scaled_y


class DecisionSet:
    """Decision set instance."""

    def __init__(self):
        self.rules = None

    def fit(self, x, y):
        """Train this instance.
        Args:
            x (ndarray): Data.
            y (ndarray): Labels.
        Returns:
            (DecisionSet): Returns `self`.
        """
        return None


def decision_tree(x, y):
    """Train a Decision Tree Classifier on x and y.
    Arguments:
        x (ndarray): Data.
        y (ndarray): Labels.
    Returns:
        (list): List of `Rule`.
    """
    dt = DecisionTreeClassifier()
    dt.fit(x, y)
    rules = _rules_from_dt(dt)

    return rules


def pruned_decision_tree(x, y, max_depth=4):
    """Train a Decision Tree Classifier on `x` and `y` with maximum depth `max_depth`.
    Arguments:
        x (ndarray): Data.
        y (ndarray): Labels.
    Returns:
        (list): List of `Rule`.
    """
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(x, y)
    rules = _rules_from_dt(dt)

    return rules


def cpar(x, y, path=None, cwd=None):
    """Train a CPAR on data x and labels y.
    Arguments:
        x (ndarray): Data.
        y (ndarray): Labels.
    Returns:
        (list): The CPAR rules.
    """
    if path is None:
        path = os.getcwd()
    if cwd is None:
        cwd = os.getcwd()

    df = DataFrame(hstack((x, y.reshape(-1, 1)))).convert_objects(convert_numeric=True)
    model = _cpar(df, path, cwd)
    rules = _rules_from_cpar(model)

    return rules


def foil(x, y, path=None, cwd=None):
    """Train a FOIL on data x and labels y.
    Arguments:
        x (ndarray): Data.
        y (ndarray): Labels.
    Returns:
        (list): The FOIL rules.
    """
    if path is None:
        path = os.getcwd()
    if cwd is None:
        cwd = os.getcwd()

    df = DataFrame(hstack((x, y.reshape(-1, 1)))).convert_objects(convert_numeric=True)
    model = _foil(df, path, cwd)
    rules = _rules_from_foil(model)

    return rules


def anchors(x, y, oracle):
    """ANCHORS adversary.
    Args:
        x (ndarray): Data.
        y (ndarray): Labels.
    Returns:
        (list): List of Anchors rules.
    """
    # List of class values to map to integers
    class_names = [0, 1]
    # List of features, class excluded
    feature_names = range(x.shape[1])
    # binary
    values_per_feature = [sorted(set(x[:, k])) for k in feature_names]
    binary_features = [k for k in feature_names if values_per_feature[k] == [0, 1]]
    categorical_names = {k: values_per_feature[k] for k in binary_features}
    categorical_names = {}

    # Constructor with class names, feature names and development data
    explainer = AnchorTabularExplainer(class_names, feature_names, x, categorical_names)
    explainer.fit(x, y, x, y)

    # Extract anchors
    rules = []
    for i in tqdm(range(x.shape[0])):
        explanation = explainer.explain_instance(x[i].reshape(1, -1), oracle.predict, threshold=0.95)
        rule = _rule_from_anchor(explanation)
        rules.append(rule)

    return rules


def lore(x, y, oracle):
    """LORE adversary.
    Args:
        x (ndarray): Data.
        y (ndarray): Labels.
    Returns:
        (list): List of LORE rules.
    """
    # List of class values to map to integers
    # List of features, class excluded
    df = pd.DataFrame((hstack((x, y.reshape(-1, 1)))))
    df.columns = list(map(str, range(df.shape[1])))
    class_name = df.columns[-1]

    df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataset(df, class_name)
    # Constructor with class names, feature names and development data
    explainer = LOREM(x, oracle.predict, feature_names, class_name, class_values, numeric_columns,
                      features_map, neigh_type='rndgen', categorical_use_prob=True,
                      continuous_fun_estimation=False, size=750, ocr=0.1, multi_label=False, one_vs_rest=False,
                      verbose=False, ngen=10)

    # Extract anchors
    rules = []
    for i in tqdm(range(x.shape[0])):
        explanation = explainer.explain_instance(x[i], use_weights=True, metric='euclidean')
        rule = _rule_from_lore(explanation)
        rules.append(rule)

    return rules


def trepan(x, y, oracle):
    """Train a Trepan instance.
    Args:
        x (ndarray): Data.
        y (ndarray): Labels.
        oracle (Predictor): The oracle used to grow the train data.
    Returns:
        (list): List of rules.
    """
    trepan = TREPAN(oracle)
    data = pd.DataFrame(hstack((x, y.reshape(-1, 1))))
    trepan = trepan.fit(data, max_nodes=3)
    rules = _rule_from_trepan(trepan)

    return rules


def decision_set(x, y):
    """Train a Decision Set instance.
    Args:
        x (ndarray): Data.
        y (ndarray): Labels.
    Returns:
        (list): List of rules.
    """
    decision_set  = DecisionSet()
    decision_set = decision_set.fit(x, y)
    rules = _rules_from_decision_set(decision_set)

    return rules


def corels(data, labels, undiscretize, one_hot_names):
    """BRL adversary.
    Args:
        data (str): Training data path.
        label (str): Training labels path.
        undiscretize (dict): Dictionary feature -> bins.
        one_hot_names (list): List of one-hot names.
    Returns:
        (list): List of CORELS rules.
    """
    cmd = wd + 'corels/src/corels -r 0.015 -c 2 -p 1 "' + data + '" "' + labels + '"'
    process = run_cmd(cmd, shell=True, stdout=PIPE)
    output = process.stdout.decode('utf-8').split('\n')
    rules_start = output.index('OPTIMAL RULE LIST') + 1
    rules_end = rules_start + output[rules_start:].index('')
    corels_rules = [output[k] for k in range(rules_start, rules_end)]

    with open(data, 'r') as log:
        data_file = log.read()
    features = re.findall('\{.+:.+\} ', data_file)
    feature_premises = [f[1:-2].split(':') for f in features]
    groups = [0] + [i for i in range(len(feature_premises) - 1)
                    if feature_premises[i][0] != feature_premises[i + 1][0]] + [len(feature_premises) - 1]
    categorical_feature_values = {}
    for g, group in enumerate(groups):
        if g < len(groups) - 2 and not feature_premises[groups[g + 1] + 1][1].isdigit():
            feature_name = feature_premises[groups[g] + 1][0]
            feature_cardinality = groups[g + 1] - groups[g]
            categorical_feature_values[feature_name] = [feature_premises[groups[g] + 1 + k][1] for k in
                                                        range(feature_cardinality)]
        elif g == len(groups) - 2 and not feature_premises[groups[-2] + 1][1].isdigit():
            feature_name = feature_premises[groups[-1]][0]
            feature_cardinality = groups[-1] - groups[-2]
            categorical_feature_values[feature_name] = [feature_premises[groups[g] + 1 + k][1] for k in
                                                        range(feature_cardinality)]

    rules = []
    for corels_rule in corels_rules:
        parsed_corels_rule = corels_rule.replace('{y:', '').replace('{', '').replace('}', '')\
                            .replace('(', '').replace(')', '').replace('if ', '').replace(' then', '').replace('else ', '')
        corels_premises = parsed_corels_rule.split(' ')
        output = int(corels_premises[-1])
        corels_premises = corels_premises[:-1]

        premises = {}
        # Last one is the default empty rule
        for corels_premise in corels_premises:
            corels_premise = corels_premise.replace('{', '').replace('}', '')

            # One-hot with positive value
            if ':' not in corels_premise:
                premises[int(corels_premise)] = (.5, +inf)
                continue

            feature, value = corels_premise.split(':')
            if value.isdigit() and int(value) in one_hot_names:
                feature, value = int(feature), int(value)
                idx = one_hot_names.index(feature if feature.isdigit() else feature)
                premises[idx] = tuple(undiscretize[feature][value])
            elif not value.isdigit() and feature + '=' + value in one_hot_names:
                # One-hot encoding
                idx = one_hot_names.index(feature + '=' + value)
                premises[idx] = 0.5, +inf
            else:
                idx = one_hot_names[int(feature)]
                premises[int(feature)] = tuple(undiscretize[idx][int(value)])

        rule = Rule(premises=premises, consequence=output)
        rules.append(rule)

    return rules


def bayesian_rule_lists(data, labels, undiscretize, one_hot_names):
    """SBRL adversary.
    Args:
        data (str): Training data path.
        label (str): Training labels path.
        undiscretize (dict): Dictionary feature -> bins.
        one_hot_names (list): List of one-hot names.
    Returns:
        (list): List of SBRL rules.
    """
    rule_ids, outputs, rule_strings = pysbrl.train_sbrl(data, labels, 20.0, eta=2.0, max_iters=2000, n_chains=10)

    outputs = outputs.argmax(axis=1)
    sbrl_rules = [rule_strings[i] for i in rule_ids]
    sbrl_rules = [rule for rule in sbrl_rules if rule != 'default']
    rules = []

    with open(data, 'r') as log:
        data_file = log.read()
    features = re.findall('\{.+:.+\} ', data_file)
    feature_premises = [f[1:-2].split(':') for f in features]
    groups = [0] + [i for i in range(len(feature_premises) - 1)
                    if feature_premises[i][0] != feature_premises[i + 1][0]] + [len(feature_premises) - 1]

    categorical_feature_values = {}
    for g, group in enumerate(groups):
        if g < len(groups) - 2 and not feature_premises[groups[g + 1] + 1][1].isdigit():
            feature_name = feature_premises[groups[g] + 1][0]
            feature_cardinality = groups[g + 1] - groups[g]
            categorical_feature_values[feature_name] = [feature_premises[groups[g] + 1 + k][1] for k in range(feature_cardinality)]
        elif g == len(groups) - 2 and not feature_premises[groups[-2] + 1][1].isdigit():
            feature_name = feature_premises[groups[-1]][0]
            feature_cardinality = groups[-1] - groups[-2]
            categorical_feature_values[feature_name] = [feature_premises[groups[g] + 1 + k][1] for k in
                                                 range(feature_cardinality)]

    for rule, output in zip(sbrl_rules[:-1], outputs):
        sbrl_premises = rule.replace('{', '').replace('}', '').split(',')
        premises = {}
        # Last one is the default empty rule
        for sbrl_premise in sbrl_premises:
            sbrl_premise = sbrl_premise.replace('{', '').replace('}', '')

            # One-hot with positive value
            if ':' not in sbrl_premise:
                premises[int(sbrl_premise)] = (.5, +inf)
                continue

            feature, value = sbrl_premise.split(':')
            if feature.isdigit() and value is not None:
                feature, value = int(feature), int(value)
                try:
                    premises[feature] = tuple(undiscretize[one_hot_names[feature]][value])
                except KeyError:
                    pass
            elif value is None:
                # One-hot encoding
                feature = int(feature)
                premises[feature] = 0.5, +inf
            else:
                idx = one_hot_names[int(feature)]
                premises[int(feature)] = tuple(undiscretize[idx][int(value)])

        rule = Rule(premises=premises, consequence=output)
        rules.append(rule)

    return rules


@click.command()
@click.option('-tr',     type=click.Path(exists=True),   help='Path to the training set.')
@click.option('-ts',     type=click.Path(exists=True),   help='Path to the test set.')
@click.option('-m',     '--model',      default=None,       help='Model to train. Can be either \'decision tree\','
                                                      ' \'pruned_decision_tree\', \'cpar\', \'foil\','
                                                       ' \'trepan\', \'ids\', \'anchors\'.')
@click.option('--dataset',    default=None,       help='Dataset name. Used to store results.')
@click.option('-d',     '--data',       default=None,       help='Data path. Provide when using either CORELS or SBRL.',
                type=click.Path(exists=True))
@click.option('-l',     '--labels',     default=None,       help='Labels path. Provide when using either CORELS or SBRL.',
                type=click.Path(exists=True))
@click.option('-b',     '--buckets',    default=None,       help='Buckets path. Provide when using either CORELS or SBRL.',
                type=click.Path(exists=True))
@click.option('-n',     '--names',      default=None,       help='One-hot names. Provide when using either CORELS or SBRL.',
                type=click.Path(exists=True))
@click.option('-o',     '--oracle',     default=None,       help='Oracle file, if required by the model.',
              type=click.Path(exists=True))
@click.option('-p',     '--path',       default=None,       help='Output path to store the generated ruleset.')
def cl_run(tr, ts, model, dataset, data, labels, buckets, names, oracle, path):
    if model in ('anchors', 'trepan', 'lore') and oracle is None:
        raise ValueError('No oracle provided.')

    if oracle is not None:
        if oracle.endswith('.h5'):
            black_box = load_model(oracle)
        elif oracle.endswith('.pickle'):
            with open(oracle, 'rb') as log:
                black_box = pickle.load(log)

    if names is not None:
        one_hot_names = pd.read_csv(names).values.tolist()[0]

    if buckets is not None:
        with open(buckets, 'r') as log:
            discretization = json.load(log)

    if tr is not None and model != 'corels' and model != 'sbrl':
        tr = pd.read_csv(tr, delimiter=',').convert_objects(convert_numeric=True)
        x, y = tr.values[:, :-1], tr.values[:, -1]

        if oracle is not None:
            y = black_box.predict(x).squeeze()

    if model == 'lore':
        rules = lore(x, y, black_box)
    elif model == 'anchors':
        rules = anchors(x, y, black_box)
    elif model == 'cpar':
        rules = cpar(x, y)
    elif model == 'foil':
        rules = foil(x, y)
    elif model == 'decision_tree':
        rules = decision_tree(x, y)
    elif model == 'pruned_decision_tree':
        rules = pruned_decision_tree(x, y)
    elif model == 'trepan':
        rules = trepan(x, y, black_box)
    elif model == 'sbrl':
        if data is None:
            tr = pd.read_csv(tr, delimiter=',').convert_objects(convert_numeric=True)
            ts = pd.read_csv(ts, delimiter=',').convert_objects(convert_numeric=True)
            full_data = pd.concat([tr, ts], axis='rows', ignore_index=True).dropna()

            if oracle is not None:
                y = black_box.predict(full_data.values[:, :-1]).round().squeeze()
                full_data.iloc[:, -1] = y
            output = '.' + str(datetime.datetime.now())

            dataset_to_sbrl_file(full_data, output)
            data = output + '.data.sbrl'
            labels = output + '.label.sbrl'

        rules = bayesian_rule_lists(data, labels, undiscretize=discretization, one_hot_names=one_hot_names)
    elif model == 'corels':
        if data is None:
            tr = pd.read_csv(tr, delimiter=',').convert_objects(convert_numeric=True)
            ts = pd.read_csv(ts, delimiter=',').convert_objects(convert_numeric=True)
            full_data = pd.concat([tr, ts], axis='rows', ignore_index=True)

            if oracle is not None:
                y = black_box.predict(full_data.values[:, :-1]).round().squeeze()
                full_data.iloc[:, -1] = y
            output = '.' + str(datetime.datetime.now())

            dataset_to_corels_file(full_data, output)
            data = output + '.data.corels'
            labels = output + '.label.corels'

        rules = corels(data, labels, undiscretize=discretization, one_hot_names=one_hot_names)
    else:
        raise ValueError('Unknown model: ' + str(model))

    with open(path, 'w') as log:
        print('Dumping in ' + path)
        json_rules = [rule.json() for rule in rules]
        json.dump(json_rules, log)


if __name__ == '__main__':
    cl_run()
