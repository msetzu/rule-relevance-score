"""
Evaluation module providing basic metrics to run and analyze GLocalX's results.
Two evaluators are provided, DummyEvaluator, which does not optimize performance,
and MemEvaluator, which stores previously computed measures to speed-up performance.
"""
from abc import abstractmethod
from statistics import harmonic_mean

import numpy as np
from scipy.spatial.distance import hamming

from models import Rule


def covers(rule, x):
    """Does `rule` cover c?

    Args:
        rule (Rule): The rule.
        x (numpy.np.array): The record.
    Returns:
        bool: True if this rule covers c, False otherwise.
    """
    return all([(x[feature] >= lower) & (x[feature] < upper)] for feature, (lower, upper) in rule)


def binary_fidelity(rule, x, y, evaluator=None, ids=None, default=np.nan):
    """Evaluate the goodness of unit.
    Args:
        rule (Rule): The unit to evaluate.
        x (np.array): The data.
        y (np.array): The labels.
        evaluator (Evaluator): Optional evaluator to speed-up computation.
        ids (np.array): Unique identifiers to tell each element in @patterns apart.
        default (int): Default prediction for records not covered by the unit.
    Returns:
          float: The unit's fidelity_weight
    """
    coverage = evaluator.coverage([rule], x, ids=ids).flatten()
    unit_predictions = np.array([rule.consequence for _ in range(x.shape[0] if ids is None else ids.shape[0])]).flatten()
    unit_predictions[~coverage] = default

    fidelity = 1 - hamming(unit_predictions, y[ids] if ids is not None else y) if len(y) > 0 else 0

    return fidelity


def coverage_size(rule, x):
    """Evaluate the cardinality of the coverage of unit on c.

    Args:
        rule (Rule): The rule.
        x (np.array): The validation set.

    Returns:
        (int: Number of records of X covered by rule.
    """
    return coverage_matrix([rule], x).sum().item(0)


def coverage_matrix(rules, x, y=None, ids=None):
    """Compute the coverage of @rules over @patterns.
    Args:
        rules (Union(list, Rule)): List of rules (or single Rule) whose coverage to compute.
        x (np.array): The validation set.
        y (np.array): The labels, if any. None otherwise. Defaults to None.
        ids (np.array): Unique identifiers to tell each element in `x` apart.
    Returns:
        numpy.ndnp.array: The coverage matrix.
    """
    def premises_from(rule, include_labels=False):
        if not include_labels:
            premises = np.logical_and.reduce([[(x[:, feature] > lower) & (x[:, feature] <= upper)]
                                              for feature, (lower, upper) in rule]).squeeze()
        else:
            premises = np.logical_and.reduce([(x[:, feature] > lower) & (x[:, feature] <= upper)
                                              & (y == rule.consequence)
                                              for feature, (lower, upper) in rule]).squeeze()

        premises = np.argwhere(premises).squeeze()

        return premises

    if isinstance(rules, Rule):
        coverage_matrix_ = np.full((len(x)), False)
        hit_columns = [premises_from(rules, y is not None)]
        coverage_matrix_[tuple(hit_columns)] = True
    else:
        coverage_matrix_ = np.full((len(rules), len(x)), False)
        hit_columns = [premises_from(rule, y is not None) for rule in rules]

        for k, hits in zip(range(len(x)), hit_columns):
            coverage_matrix_[k, hits] = True

    coverage_matrix_ = coverage_matrix_[:, ids] if ids is not None else coverage_matrix_

    return coverage_matrix_


class Evaluator:
    """Evaluator interface. Evaluator objects provide coverage and fidelity_weight utilities."""

    @abstractmethod
    def coverage(self, rules, patterns, target=None, ids=None):
        """Compute the coverage of @rules over @patterns.
        Args:
            rules (list) or (Rule):
            patterns (np.array): The validation set.
            target (np.array): The labels, if any. None otherwise. Defaults to None.
            ids (np.array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            numpy.array: The coverage matrix.
        """
        pass

    @abstractmethod
    def coverage_size(self, rule, x, ids=None):
        """Evaluate the cardinality of the coverage of unit on c.

        Args:
            rule (Rule): The rule.
            x (np.array): The validation set.
            ids (np.array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            int: Number of records of X covered by rule.
        """
        pass

    @abstractmethod
    def binary_fidelity(self, unit, x, y, ids=None, default=np.nan):
        """Evaluate the goodness of unit.
        Args:
            unit (Unit): The unit to evaluate.
            x (np.array): The data.
            y (np.array): The labels.
            ids (np.array): Unique identifiers to tell each element in @patterns apart.
            default (int): Default prediction when no rule covers a record.
        Returns:
              float: The unit's fidelity_weight
        """
        pass

    @abstractmethod
    def binary_fidelity_model(self, rules, scores, x, y, x_vl, y_vl, default=None, ids=None):
        """Evaluate the goodness of the `units`.
        Args:
            rules (Union(list, set)): The units to evaluate.
            scores (np.array): The scores.
            x (np.array): The training data.
            y (np.array): The training labels.
            x_vl (np.array): The data.
            y_vl (np.array): The labels.
            default (int): Default prediction for records not covered by the unit.
            ids (np.array): Unique identifiers to tell each element in @c apart.
        Returns:
            float: The units fidelity_weight.
        """
        pass

    @abstractmethod
    def covers(self, rule, x):
        """Does @rule cover c?

        Args:
            rule (Rule): The rule.
            x (np.array): The record.
        Returns:
            bool: True if this rule covers c, False otherwise.
        """
        pass


class MemEvaluator(Evaluator):
    """Memoization-aware Evaluator to avoid evaluating the same measures over the same data."""

    def __init__(self, oracle):
        """Constructor."""
        self.oracle = oracle
        self.coverages = dict()
        self.binary_fidelities = dict()
        self.coverage_sizes = dict()
        self.scores = dict()

    @abstractmethod
    def covers(self, rule, x):
        """Does @rule cover c?

        Args:
            rule (Rule): The rule.
            x (np.array): The record.
        Returns:
            bool: True if this rule covers c, False otherwise.
        """
        return covers(rule, x)

    def coverage(self, rules, x, y=None, ids=None):
        """Compute the coverage of @rules over @patterns.
        Args:
            rules (Union(Rule, list): Rule (or list of rules) whose coverage to compute.
            x (np.array): The validation set.
            y (np.array): The labels, if any. None otherwise. Defaults to None.
            ids (np.array): IDS of the given `patterns`, used to speed up evaluation.
        Returns:
            numpy.np.array: The coverage matrix.
        """
        for rule in rules:
            if rule not in self.coverages:
                self.coverages[rule] = coverage_matrix(rule, x, y)
        cov = np.array([self.coverages[rule] for rule in rules])
        cov = cov[:, ids] if ids is not None else cov

        return cov

    def binary_fidelity(self, unit, x, y, default=np.nan, ids=None):
        """Evaluate the goodness of unit.
        Args:
            unit (Unit): The unit to evaluate.
            x (np.array): The data.
            y (np.array): The labels.
            default (int): Default prediction for records not covered by the unit.
            ids (np.array): IDS of the given `x`, used to speed up evaluation.
        Returns:
              float: The unit's fidelity_weight
        """
        if y is None:
            y = self.oracle.predict(x).round().squeeze()

        if ids is None:
            self.binary_fidelities[unit] = self.binary_fidelities.get(unit, binary_fidelity(unit, x, y, self,
                                                                                            default=default, ids=None))
            fidelity = self.binary_fidelities[unit]
        else:
            fidelity = binary_fidelity(unit, x, y, self, default=default, ids=ids)

        return fidelity

    def binary_fidelity_model(self, rules, scores, x, y, x_vl, y_vl, default=None, ids=None):
        """Evaluate the goodness of unit.
        Args:
            rules (np.array): The units to evaluate.
            scores (np.array): The scores.
            x (np.array): The data.
            y (np.array): The labels.
            x_vl (np.array): The validation data.
            y_vl (np.array): The validation labels.
            default (int): Default prediction for records not covered by the unit.
            ids (np.array): Unique identifiers to tell each element in @patterns apart.
        Returns:
              (float): The units fidelity.
        """
        if self.oracle is not None:
            y_vl = self.oracle.predict(x_vl).round().squeeze()

        coverage = self.coverage(rules, x_vl, y_vl)

        predictions = []
        for record in range(len(x_vl)):
            companions = scores[coverage[:, record]]
            companion_units = [rules[i] for i in coverage[:, record]]
            top_companions = np.argsort(companions)[-1:]
            top_units = [companion_units[i] for i in top_companions]
            top_fidelities = companions[top_companions]
            top_fidelities_0 = [top_fidelity for top_fidelity, top_unit in zip(top_fidelities, top_units)
                                if top_unit.consequence == 0]
            top_fidelities_1 = [top_fidelity for top_fidelity, top_unit in zip(top_fidelities, top_units)
                                if top_unit.consequence == 1]

            if len(top_fidelities_0) == 0 and len(top_fidelities_1) > 0:
                prediction = 1
            elif len(top_fidelities_0) > 0 and len(top_fidelities_1) == 0:
                prediction = 0
            elif len(top_fidelities_0) == 0 and len(top_fidelities_1) == 0:
                prediction = default
            else:
                prediction = 0 if np.mean(top_fidelities_0) > np.mean(top_fidelities_1) else 1

            predictions.append(prediction)
        predictions = np.array(predictions)
        fidelity = 1 - hamming(predictions, y_vl) if len(y_vl) > 0 else 0

        return fidelity


########################
# Framework validation #
########################
def validate(rules, scores, oracle, vl, scoring='rrs', evaluator=None, alpha=0, beta=0, gamma=-1, max_len=np.inf):
    """Validate the given rules in the given ruleset.
    Arguments:
        rules (iterable): Iterable of rules to validate.
        scores(iterable): The scores.
        oracle (Predictor): Oracle to validate against.
        vl (np.array): Validation set.
        scoring (str): Scoring function used.
        evaluator (Evaluator): Evaluator.
        alpha (float): Pruning hyperparameter, rules with score less than `alpha` are removed from the ruleset used to
                        perform the validation.
        beta (float): Pruning hyperparameter, rules with score less than the `beta`-percentile are removed from the
                        result.
        gamma (int): Maximum number of rules to use.
        max_len (int): Pruning hyperparameter, rules with length
                        more than `max_len` are removed from the
                        ruleset used to perform the validation.
    Returns:
        (dict): Dictionary of validation measures.
    """
    def mean_len(rules):
        return np.mean([len(r) for r in rules])

    def std_len(rules):
        return np.std([len(r) for r in rules])

    def len_reduction(ruleset_a, ruleset_b):
        return ruleset_a / ruleset_b

    def simplicity(ruleset_a, ruleset_b):
        return ruleset_a * ruleset_b

    def features_frequencies(rules, features):
        return [sum([1 if f in r else 0 for r in rules]) for f in features]

    def coverage_pct(rules, x):
        coverage = coverage_matrix(rules, x)
        coverage_percentage = (coverage.sum(axis=0) > 0).sum() / x.shape[0]

        return coverage_percentage

    if evaluator is None:
        evaluator = MemEvaluator(oracle=oracle)
    if oracle is None:
        x = vl[:, :-1]
        y = vl[:, -1]
    else:
        x = vl[:, :-1]
        y = oracle.predict(x).round().squeeze()

    # Prune rules according to score and maximum length
    majority_label = round(y.sum() / len(y))
    n = len(rules)

    beta_value = np.percentile(scores, beta)
    beta_score_idx = [i for i in range(n) if scores[i] >= beta_value and len(rules[i]) <= max_len][:gamma]
    beta_rules = [rules[i] for i in beta_score_idx]
    beta_rules_nr = len(beta_rules)

    alpha_score_idx = [i for i in range(n) if scores[i] >= alpha and len(rules[i]) <= max_len][:gamma]
    alpha_rules = [rules[i] for i in alpha_score_idx]

    validation = dict()
    validation['alpha']     = alpha
    validation['beta']      = beta
    validation['gamma']     = gamma
    validation['max_len']   = max_len
    validation['scoring']   = scoring

    scoring_fidelities = evaluator.binary_fidelity_model(alpha_rules, scores=scores,
                                                                x=x, y=y, x_vl=x, y_vl=y,
                                                                default=majority_label)
    beta_scoring_fidelities = evaluator.binary_fidelity_model(beta_rules,
                                                               scores=scores[beta_score_idx],
                                                               x=x, y=y, x_vl=x, y_vl=y,
                                                               default=majority_label)

    validation['scoring-fidelities'] = scoring_fidelities
    validation['beta-scoring-fidelity'] = beta_scoring_fidelities
    validation['beta-scoring-rule_nr'] = beta_rules_nr
    validation['mean_length'] = mean_len(rules)
    validation['std_length'] = std_len(rules)

    validation['coverage'] = coverage_pct(rules, x)
    validation['beta-scoring-coverage'] = coverage_pct(beta_rules, x)
    validation['mean-beta-scoring-length'] = mean_len(beta_rules)
    validation['std-beta-scoring-length'] = std_len(beta_rules)

    # Predictions
    validation['mean_prediction'] = np.mean([r.consequence for r in rules])
    validation['std-scoring-prediction'] = np.std([r.consequence for r in alpha_rules])
    validation['rule_reduction-alpha-scoring'] = len(alpha_rules) / len(rules)
    validation['rule_reduction-beta-scoring'] = len(beta_rules) / len(rules)
    validation['len_reduction-beta-scoring'] = len_reduction(validation['mean-beta-scoring-length'],
                                                             validation['mean_length'])
    validation['simplicity-beta-scoring'] = simplicity(validation['rule_reduction-beta-scoring'],
                                                       validation['len_reduction-beta-scoring'])
    features = rules[0].names
    validation['feature_frequency']                 = features_frequencies(rules, features)
    validation['scoring-beta-feature_frequency']    = features_frequencies(beta_rules, features)

    validation['scoring-beta-escore'] = harmonic_mean([validation['beta-scoring-fidelity'],
                                                        validation['simplicity-beta-scoring']])

    return validation
