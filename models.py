"""Models to score, mainly rules. A model should implement the `Unit` interface."""
import json
from copy import deepcopy
from math import floor

from scipy.spatial.distance import euclidean
from numpy import sign, argwhere, inf, array

from eval import coverage_matrix

__all__ = ['Unit', 'Vector', 'Rule', 'centrality']


class Unit:
    """General computational unit with a `predict` method."""

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return 'None'

    def __str__(self):
        return 'Empty'

    def predict(self, x):
        """Evaluate any.

        Args:
            x (object): The object to evaluate.
        Returns:
              None
        """
        return None


class Vector(Unit):
    """Vector unit."""

    def __init__(self, vector, position, distance=euclidean):
        self.__vector = vector
        self.__dimension = len(vector)
        self.__position = position
        self.__eval = distance

    def __len__(self):
        return self.__dimension

    def __hash__(self):
        return hash(self.__vector)

    def __getitem__(self, item):
        return self.__vector[item]

    def __add__(self, other):
        sum_ = self.__vector + other.__vector
        return sum_

    def __sub__(self, other):
        sub_ = self.__vector - other.__vector
        return sub_

    def __mul__(self, other):
        mul_ = self.__vector * other.__vector
        return mul_

    def __repr__(self):
        return repr(self.__vector) + '\n' \
               + repr(self.__dimension) + '\n' \
               + repr(self.__position) + '\n'

    def __str__(self):
        return 'Vector: ' + repr(self.__vector) + '\n' \
               + 'Dimensionality: ' + repr(self.__dimension) + '\n' \
               + 'Position: ' + repr(self.__position) + '\n'

    def predict(self, x):
        return self.__eval(x, self.__vector)


def centrality(v, l, u, distance):
    """
    Compute the centrality of the given value @v w.r.t. the range (@l, @u).

    Args:
        v (float): The value whose centrality index to compute.
        l (float): The lower bound.
        u (float): The upper bound.
        distance (function): The distance function. Should be of the form
                                float, float -> float.
    Returns:
        (float): Centrality index of @v.
    """
    lower_centrality = distance(v, l)
    upper_centrality = distance(v, u)
    max_ = max(lower_centrality, upper_centrality)
    min_ = min(lower_centrality, upper_centrality)
    score = min_ / max_ if max_ != 0 else 1

    return score


class Rule(Unit):
    """A logical rule in CNF form."""

    def __init__(self, premises=None, consequence=None, invert=None, distance=euclidean):
        """
        Default rule with the given premises and consequence.

        Args:
            premises (dict): Dictionary {feature -> premise} holding the premise for @feature. Defaults to the empty
                                dictionary.
            consequence (int): Outcome of the rule. Defaults to None.
            invert (function): Function consequence -> consequence to invert the rule's consequence.
                                Defaults to (consequence + 1) % 2
            distance (function): Function Rule x Record -> float to evaluate the centrality of a record
                                w.r.t. the rule. Defaults to Euclidean distance.
        """
        self.features = set(premises.keys())
        self.premises = premises if premises is not None else dict()
        self.consequence = consequence
        self._invert_consequence = invert if invert is not None else lambda c: -1
        self.distance = distance

    @classmethod
    def fromarrays(cls, features=None, thresholds=None, consequence=None, path=None):
        """
        Arguments:
            features (list): The list of features.
            thresholds (list): The list of thresholds per node in the path.
            consequence (int): Rule consequence.
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

        thresholds_ = thresholds[:-1]
        indices_per_feature = {feature: argwhere(__features == feature).flatten()
                               for feature in __features}
        directions_per_feature = {f: [sign(path[k + 1]) for k in indices_per_feature[f] if k < len(path) - 1]
                                  for f in __features[:-1]}

        for feature in __features:
            if len(indices_per_feature[feature]) == 1:
                threshold = thresholds_[indices_per_feature[feature][0]]
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
        return len(self.premises)

    def __repr__(self):
        return str(self.premises) + '\n' + str(self.consequence)

    def __str__(self):
        str_ = '{\n'
        for k in sorted(self.features):
            str_ = str_ + '\t' + str(k) + ': ' + str(self.premises[k]) + '\n'
        str_ += str(self.consequence)
        str_ += '}\n'

        return str_

    def __eq__(self, other):
        """ True iff the @other has the same ranges of this. """
        return self.premises == other.premises and self.consequence == other.consequence

    def __hash__(self):
        return hash(tuple([(k, v1, v2) for k, (v1, v2) in self.premises.items()])) ^ hash(self.consequence)

    def __contains__(self, item):
        return item in self.premises

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
        cop_rule = Rule(self.premises, self.consequence, invert=self._invert_consequence)

        return cop_rule

    def __deepcopy__(self, memodict=None):
        cop_rule = Rule(deepcopy(self.premises), deepcopy(self.consequence), invert=deepcopy(self._invert_consequence))

        return cop_rule

    def json(self):
        """
        Encode this object in a JSON string. Note that this string is equivalent to str(self).

        Returns:
            (dict): The JSON dictionary representation of this object.
        """
        json_dict = {int(k): [v1, v2] for k, (v1, v2) in self.premises.items()}
        json_dict['consequence'] = int(self.consequence)

        return json_dict

    @classmethod
    def from_json(cls, json_file):
        """
        Read rules from json_file.

        Args:
            json_file (str): Path of the json file.
        Returns:
            (list): List of rules in json_file.
        """
        with open(json_file, 'r') as log:
            jsonized_rules = json.load(log)
            premises = [{int(k): v for k, v in dic.items() if k != 'consequence' and k != 'label'} for dic in jsonized_rules]
            consequences = [dic['consequence'] if 'consequence' in dic else dic['label'] for dic in jsonized_rules]
        rules = [Rule(premises=premise, consequence=consequence)
                 for premise, consequence in zip(premises, consequences)]

        return rules

    def to_dict(self):
        """
        Compute the python dictionary associated with this rule.

        Returns:
            (dict): Python dictionary.
        """
        this_copy = deepcopy(self)
        this_copy.premises['consequence'] = self.consequence

        return this_copy

    def __invert__(self):
        """Negate rule by swapping its consequence. Defaults to (consequence + 1) % 2 if
        @invert was not provided at construction time.

        Returns:
            (Rule): Rule with the same premises and inverted consequence.
        """
        neg_rule = deepcopy(self)
        neg_rule.consequence = self._invert_consequence(self.consequence)

        return neg_rule

    def __add__(self, other):
        """Sum to rule according to the quasi-polyhedra union.

        Args:
            other (Rule): The rule to add.

        Returns:
            (Rule): New rule with united premises and same consequence.
                    Throws ValueError when discordant consequences are found.
        """
        if self.consequence != other.consequence:
            raise ValueError('Rules should have the same consequence')

        sum_rule = Rule({}, self.consequence, invert=self._invert_consequence)
        premises_in_common = {feature for feature in self.premises if feature in other.premises}
        premises_exclusive = {feature for feature in self.premises if feature not in other.premises}

        for f in premises_in_common:
            sum_rule[f] = (min(self[f][0], other[f][0]), max(self[f][1], other[f][1]))
        for f in premises_exclusive:
            sum_rule[f] = self[f]
        sum_rule.features = set(sum_rule.premises.keys())

        return sum_rule

    def __sub__(self, other):
        """Subtract to rule according to the quasi-polyhedra union.

        Args:
            other (Rule): The rule to subtract.

        Returns:
            (Rule): New rule with different premises and same consequence as self.
        """
        premises_in_common = self.features & other.features

        negated_premises = {}
        additional_rules_nr = 0
        for f in premises_in_common:
            self_a, self_b, other_a, other_b = self[f][0], self[f][1], other[f][0], other[f][1]

            # weak included in strong
            if self_a <= other_a <= other_b <= self_b:
                negated_premises[f] = None
            # weak shifted to the left
            elif other_a < self_a < other_b <= self_b:
                negated_premises[f] = (other_a, self_a)
            # weak shifted to the right
            elif self_a <= other_a <= self_b < other_b:
                negated_premises[f] = (self_b, other_b)
            # strong included in the weak, split the weak in two
            elif other_a < self_a <= self_b < other_b:
                negated_premises[f] = [(other_a, self_a), (self_b, other_b)]
                additional_rules_nr += 1

        new_rules = [Rule(premises={}, consequence=other.consequence, invert=self._invert_consequence)
                     for _ in range(additional_rules_nr + 1)]

        # Preserve features in the weak rule, but not in the strong
        for f, val in other:
            if f not in negated_premises:
                for rule in new_rules:
                    rule[f] = val

        # Cut according to the negated submissive ranges
        for f in negated_premises.keys():
            if f in negated_premises:
                if negated_premises[f] is None:
                    for rule in new_rules:
                        if f in rule:
                            del rule[f]
                elif isinstance(negated_premises[f], list):
                    # First half of the rules with first value
                    for rule in new_rules[:floor(additional_rules_nr / 2)]:
                        rule[f] = negated_premises[f][0]
                    # Second half of the rules with second value
                    for rule in new_rules[floor(additional_rules_nr / 2):]:
                        rule[f] = negated_premises[f][1]
                else:
                    for rule in new_rules:
                        rule[f] = negated_premises[f]

        new_rules = set(new_rules)

        return new_rules

    def covers(self, x):
        """Does this rule cover x?

        Args:
            x (array): The record.
        Returns:
            (boolean): True if this rule covers x, False otherwise.
        """
        return all([[(x[feature] >= lower) & (x[feature] < upper)] for feature, (lower, upper) in self])

    def predict(self, x, evaluator=None, indices=None):
        """Predict x with a level of uncertainty.
        Args:
            x (array): Value to predict.
            evaluator (Evaluator): Optional evaluator to speed-up computation.
            indices (array): Unique identifiers to tell each element in @x apart.
        Returns:
              (int): Value in {0, 1}, prediction of the rule on x.
        """
        if evaluator is None:
            predictions = array([self.consequence for _ in range(x.shape[0])])
            coverage = coverage_matrix([self], x)
            predictions[~coverage] = (~self).consequence
        else:
            predictions = array([self.consequence for _ in range(x.shape[0])])
            coverage = evaluator.coverage(self, x, ids=indices)
            predictions[~coverage] = (~self).consequence

        return predictions
