"""Models to score rules."""
import json
from copy import deepcopy

from scipy.spatial.distance import euclidean
from numpy import sign, argwhere, inf, array


class Rule:
    """A_ logical rule in conjunctive form."""

    def __init__(self, premises=None, consequence=None, distance=euclidean, names=None):
        """
        Default rule with the given premises and consequence.

        Args:
            premises (dict): Dictionary {feature -> premise} holding the premise for @feature. Defaults to the empty
                                dictionary.
            consequence (int): Outcome of the rule. Defaults to None.
            distance (function): Function Rule x Record -> float to evaluate the centrality of a record
                                w.r.t. the rule. Defaults to Euclidean distance.
            names (list): Names of categorical variables. Set of sets, each set is a group of features on the same
            variable.
        """
        self.features = set(premises.keys())
        self.premises = premises if premises is not None else dict()
        self.consequence = consequence
        self.distance = distance
        self.names = names

    @classmethod
    def fromarrays(cls, features=None, thresholds=None, consequence=None, path=None):
        """
        Args:
            features (list): The list of features.
            thresholds (list): The list of thresholds per node in the path.
            consequence (int): Rules consequences.
            path (list): Path followed by the rules.
        """
        if thresholds is None:
            raise ValueError('Missing thresholds')
        if consequence is None:
            raise ValueError('Missing consequence')
        if path is None:
            raise ValueError('Missing path')
        if features is None:
            raise ValueError('Missing features')

        __premises = {}
        __features = features
        __consequence = int(consequence)

        # thresholds_ = thresholds[:-1]
        indices_per_feature = {feature: argwhere(__features == feature).flatten()
                               for feature in __features}
        directions_per_feature = {f: [sign(path[k + 1]) for k in indices_per_feature[f] if k < len(path) - 1]
                                  for f in __features}

        for feature in __features:
            if len(indices_per_feature[feature]) == 1:
                try:
                    threshold = thresholds[indices_per_feature[feature][0]]
                except IndexError as e:
                    raise e
                __premises[feature] = (-inf, threshold) if directions_per_feature[feature][0] < 0 else (threshold, inf)
            else:
                lower_bounds_idx = [index for index, direction in zip(indices_per_feature[feature],
                                                                      directions_per_feature[feature])
                                    if direction > 0]
                upper_bounds_idx = [index for index, direction in zip(indices_per_feature[feature],
                                                                      directions_per_feature[feature])
                                    if direction < 0]
                lower_bounds, upper_bounds = (array([thresholds[lower_idx] for lower_idx in lower_bounds_idx]),
                                              array([thresholds[upper_idx] for upper_idx in upper_bounds_idx]))

                if lower_bounds.shape[0] > 0 and upper_bounds.shape[0] > 0:
                    __premises[feature] = (max(lower_bounds), min(upper_bounds))
                elif lower_bounds.shape[0] == 0:
                    __premises[feature] = (-inf, min(upper_bounds))
                elif upper_bounds.shape[0] == 0:
                    __premises[feature] = (max(lower_bounds), +inf)

        return cls(__premises, __consequence)

    def __len__(self):
        if not hasattr(self, 'names') or self.names is None:
            rule_len = len(self.premises)
        else:
            # Categorical names have the form 'variable=value', hence create a set of 'variable' prefixes
            features_names = set([self.names[f].split('=')[0] if '=' in self.names[f] else self.names[f]
                                  for f in self.premises.keys()])
            rule_len = len(features_names)

        return rule_len

    def __repr__(self):
        return str(self.premises) + '\n' + str(self.consequence)

    def __str__(self):
        str_ = '{\n'
        for k in sorted(self.features):
            feature_name = self.names[k]
            str_ = str_ + '\t' + feature_name + ': ' + str(self.premises[k]) + '\n'
        str_ += '\n\tlabel: ' + str(self.consequence) + '\n'
        str_ += '}\n'

        return str_

    def __eq__(self, other):
        """True iff the `other` has the same ranges of this."""
        return self.premises == other.premises and self.consequence == other.consequence

    def __hash__(self):
        return hash(tuple([(k, v1, v2, self.consequence) for k, (v1, v2) in self.premises.items()]))

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self.premises
        elif isinstance(item, str):
            names = [self.names[p] for p in self.premises]
            return item in names
        else:
            return False

    def __getitem__(self, item):
        if item not in self.premises:
            raise KeyError(str(item) + ' not in rule')
        return self.premises[item]

    def __setitem__(self, key, value):
        self.premises[key] = value
        return self

    def __iter__(self):
        for el in self.premises.items():
            yield el

    def __delitem__(self, key):
        del self.premises[key]
        return self

    def __copy__(self):
        cop_rule = Rule(self.premises, self.consequence)

        return cop_rule

    def __deepcopy__(self, memodict=None):
        cop_rule = Rule(deepcopy(self.premises), deepcopy(self.consequence))

        return cop_rule

    def json(self):
        """
        Encode this object in a JSON dictionary.

        Returns:
            dict: The JSON dictionary representation of this object.
        """
        json_dict = {int(k): [v1, v2] for k, (v1, v2) in self.premises.items()}
        json_dict['label'] = int(self.consequence)

        return json_dict

    @classmethod
    def from_json(cls, json_file, names=None):
        """
        Read rules from json_file.

        Args:
            json_file (str): Path of the json file.
            names (list): Features names, if any, None otherwise. Defaults to None.
        Returns:
            list: List of rules in json_file.
        """
        with open(json_file, 'r') as log:
            jsonized_rules = json.load(log)
            premises = [{int(k): v for k, v in dic.items() if k != 'consequence' and k != 'label'}
                        for dic in jsonized_rules]
            consequences = [dic['consequence'] if 'consequence' in dic else dic['label'] for dic in jsonized_rules]
        rules = [Rule(premises=premise, consequence=consequence)
                 for premise, consequence in zip(premises, consequences)]
        if names is not None:
            for r in rules:
                r.names = names

        return rules

    def to_dict(self):
        """
        Compute the python dictionary associated with this rule.

        Returns:
            dict: Python dictionary.
        """
        this_copy = deepcopy(self)
        this_copy.premises['label'] = self.consequence

        return this_copy

    def __invert__(self):
        """Negate rule by swapping its consequence. Defaults to (consequence + 1) % 2 if
        @invert was not provided at construction time.

        Returns:
            Rule: Rule with the same premises and inverted consequence.
        """
        neg_rule = deepcopy(self)
        neg_rule.consequence = (self.consequence + 1) % 2

        return neg_rule
