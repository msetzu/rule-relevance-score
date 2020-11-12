"""
Evaluation module providing basic metrics to run and analyze SOME's results.
Two evaluators are provided, DummyEvaluator, which does not optimize performance,
and MemEvaluator, which stores previously computed measures to speed-up performance.
"""
from abc import abstractmethod
from statistics import harmonic_mean

from numpy import logical_and, argwhere, array, argsort, mean, full, vectorize, flip, std, inf, nan, percentile, \
    median
from scipy.spatial.distance import hamming

import numpy as np
__all__ = ['Evaluator', 'MemEvaluator', 'DummyEvaluator', 'coverage_matrix', 'binary_fidelity', 'validate']


def covers(rule, x):
    """Does `rule` cover `x`?

    Args:
        rule (Rule): The rule.
        x (numpy.array): The record.
    Returns:
        (boolean): True if this rule covers c, False otherwise.
    """
    return all([(x[feature] >= lower) & (x[feature] < upper)] for feature, (lower, upper) in rule)


def binary_fidelity(unit, x, y, evaluator=None, ids=None, default=nan):
    """Evaluate the goodness of unit.
    Args:
        unit (Unit): The unit to evaluate.
        x (array): The data.
        y (array): The labels.
        evaluator (Evaluator): Optional evaluator to speed-up computation.
        ids (array): Unique identifiers to tell each element in `x` apart.
        default (int): Default prediction for records not covered by the unit.
    Returns:
          (float): The unit's fidelity
    """
    coverage = evaluator.coverage(unit, x, ids=ids).flatten()
    unit_predictions = array([unit.consequence for _ in range(x.shape[0])]).flatten()
    unit_predictions[~coverage] = default

    fidelity = 1 - hamming(unit_predictions, y) if len(y) > 0 else 0

    return fidelity


def coverage_size(rule, x):
    """Evaluate the cardinality of the coverage of unit on `x`.

    Args:
        rule (Rule): The rule.
        x (array): The validation set.

    Returns:
        (int): Number of records of X covered by rule.
    """
    return coverage_matrix([rule], x).sum().item(0)


def coverage_matrix(rules, x, y=None):
    """Compute the coverage of @rules over @patterns.
    Args:
        rules (Union(list, Rule)): List of rules (or single Rule) whose coverage to compute.
        x (array): The validation set.
        y (array): The labels, if any. None otherwise. Defaults to None.
    Returns:
        (array): The coverage matrix.
    """
    def premises_from(rule, pure=False):
        if not pure:
            premises = logical_and.reduce([[(x[:, feature] > lower) & (x[:, feature] <= upper)]
                                           for feature, (lower, upper) in rule]).squeeze()
        else:
            premises = logical_and.reduce([(x[:, feature] > lower) & (x[:, feature] <= upper)
                                           & (y == rule.consequence)
                                           for feature, (lower, upper) in rule]).squeeze()

        premises = argwhere(premises).squeeze()

        return premises

    if isinstance(rules, list):
        coverage_matrix_ = full((len(rules), len(x)), False)
        hit_columns = [premises_from(rule, y is not None) for rule in rules]

        for k, hits in zip(range(len(x)), hit_columns):
            coverage_matrix_[k, hits] = True
    else:
        coverage_matrix_ = full((len(x)), False)
        hit_columns = [premises_from(rules, y is not None)]
        coverage_matrix_[hit_columns] = True

    return coverage_matrix_


class Evaluator:
    """Evaluator interface. Evaluator objects provide coverage and fidelity utilities."""

    @abstractmethod
    def coverage(self, rules, patterns, target=None, ids=None):
        """Compute the coverage of @rules over @patterns.
        Args:
            rules (list) or (Rule):
            patterns (array): The validation set.
            target (array): The labels, if any. None otherwise. Defaults to None.
            ids (array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            (array): The coverage matrix.
        """
        pass

    @abstractmethod
    def coverage_size(self, rule, x, ids=None):
        """Evaluate the cardinality of the coverage of unit on c.

        Args:
            rule (Rule): The rule.
            x (array): The validation set.
            ids (array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            (int): Number of records of X covered by rule.
        """
        pass

    @abstractmethod
    def binary_fidelity(self, unit, x, y, ids=None):
        """Evaluate the goodness of unit.
        Args:
            unit (Unit): The unit to evaluate.
            x (array): The data.
            y (array): The labels.
            ids (array): Unique identifiers to tell each element in @patterns apart.
        Returns:
              (float): The unit's fidelity
        """
        pass

    @abstractmethod
    def binary_fidelity_model(self, units, scores, x, y, x_vl, y_vl, default=None, ids=None):
        """Evaluate the goodness of unit.
        Args:
            units (array): The units to evaluate.
            scores (iterable): Scores to use to select the top-k units.
            x (array): The data.
            y (array): The labels.
            x_vl (array): The validation data.
            y_vl (array): The validation labels.
            default (int): Default prediction for records not covered by the unit.
            ids (array): Unique identifiers to tell each element in @patterns apart.
        Returns:
              (float): The units fidelity.
        """
        pass

    @abstractmethod
    def score(self, unit, data, target, ids=None):
        """Score @unit against @data with labels @target.
        Args:
            unit (Unit): Unit to score.
            data (array): Data.
            target (array): Targets.
            ids (array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            (float): Score.
        """
        pass


class DummyEvaluator(Evaluator):
    """Dummy evaluator with no memory: every result is computed at each call!"""

    def __init__(self, oracle):
        """Constructor."""
        self.oracle = oracle
        self.coverages = dict()
        self.binary_fidelities = dict()
        self.coverage_sizes = dict()

    def coverage(self, rules, patterns, target=None, ids=None):
        """Compute the coverage of @rules over @patterns.
        Args:
            rules (Union(Rule, list): Rule (or list of rules) whose coverage to compute.
            patterns (array): The validation set.
            target (array): The labels, if any. None otherwise. Defaults to None.
            ids (array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            (array): The coverage matrix.
        """
        rules_ = rules if isinstance(rules, list) else [rules]
        coverage_ = coverage_matrix(rules_, patterns, target)

        return coverage_

    def coverage_size(self, rule, x, ids=None):
        """Evaluate the cardinality of the coverage of unit on c.

        Args:
            rule (Rule): The rule.
            x (array): The validation set.
            ids (array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            (int): Number of records of X covered by rule.
        """
        if rule not in self.coverage_sizes:
            self.coverage_sizes[rule] = coverage_size(rule, x)

        return self.coverage_sizes[rule]

    def binary_fidelity(self, unit, x, y, ids=None, default=nan):
        """Evaluate the goodness of unit.
        Args:
            unit (Unit): The unit to evaluate.
            x (array): The data.
            y (array): The labels.
            ids (array): Unique identifiers to tell each element in @patterns apart.
            default (int): Default prediction when no rule covers a record.
        Returns:
              (float): The unit's fidelity
        """
        if self.oracle is not None or y is None:
            y = self.oracle.predict(x).squeeze()

        return binary_fidelity(unit, x, y, self, default=default)

    def binary_fidelity_model(self, units, scores, x, y, x_vl, y_vl, default=None, ids=None):
        """Evaluate the goodness of unit.
        Args:
            units (array): The units to evaluate.
            scores (array): The scores.
            x (array): The data.
            y (array): The labels.
            x_vl (array): The validation data.
            y_vl (array): The validation labels.
            default (int): Default prediction for records not covered by the unit.
            ids (array): Unique identifiers to tell each element in @patterns apart.
        Returns:
              (float): The units fidelity.
        """
        if self.oracle is not None:
            y_vl = self.oracle.predict(x_vl).round().squeeze()

        coverage = self.coverage(units, x_vl, y_vl)

        predictions = []
        for record in range(len(x_vl)):
            companions = scores[coverage[:, record]]
            companion_units = units[coverage[:, record]]
            top_companions = argsort(companions)[-1:]
            top_units = companion_units[top_companions]
            top_fidelities = companions[top_companions]
            top_fidelities_0 = [top_fidelity for top_fidelity, top_unit in zip(top_fidelities, top_units)
                                if top_unit.consequence == 0]
            top_fidelities_1 = [top_fidelity for top_fidelity, top_unit in zip(top_fidelities, top_units)
                                if top_unit.consequence == 1]

            if len(top_fidelities_0) == 0 and len(top_fidelities_1) > 0:
                prediction = 1
            elif len(top_fidelities_1) == 0 and len(top_fidelities_0) > 0:
                prediction = 0
            elif len(top_fidelities_1) == 0 and len(top_fidelities_0) == 0:
                prediction = default
            else:
                prediction = 0 if mean(top_fidelities_0) > mean(top_fidelities_1) else 1

            predictions.append(prediction)
        predictions = array(predictions)
        fidelity = 1 - hamming(predictions, y_vl) if len(y_vl) > 0 else 0

        return fidelity

    def score(self, unit, data, target, ids=None):
        """Score @unit against @data with labels @target.
        Args:
            unit (Unit): Unit to score.
            data (array): Data.
            target (array): Targets.
            ids (array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            (float): Score.
        """
        if self.oracle is not None or target is None:
            target = self.oracle.predict(data).squeeze()
        default = round(target.sum() / len(target))

        return self.binary_fidelity(unit, data, target, default=default)


class MemEvaluator(Evaluator):
    """Memoization-aware Evaluator to avoid evaluating the same measures over the same data."""

    def __init__(self, oracle):
        """Constructor."""
        self.oracle = oracle
        self.coverages = dict()
        self.binary_fidelities = dict()
        self.coverage_sizes = dict()
        self.scores = dict()

    def coverage(self, rules, patterns, targets=None, ids=None):
        """Compute the coverage of @rules over @patterns.
        Args:
            rules (Union(Rule, list): Rule (or list of rules) whose coverage to compute.
            patterns (array): The validation set.
            targets (array): The labels, if any. None otherwise. Defaults to None.
            ids (array): IDS of the given `patterns`, used to speed up evaluation.
        Returns:
            (array): The coverage matrix.
        """
        rules_ = rules if isinstance(rules, list) else [rules]
        coverage_ = coverage_matrix(rules_, patterns, targets)

        return coverage_

    def coverage_size(self, rule, x, ids=None):
        """Evaluate the cardinality of the coverage of unit on c.

        Args:
            rule (Rule): The rule.
            x (array): The validation set.
            ids (array): Unique identifiers to tell each element in @c apart.

        Returns:
            (int): Number of records of X covered by rule.
        """
        if rule not in self.coverage_sizes:
            self.coverage_sizes[rule] = {}
            for x_, i in zip(x, ids):
                coverage = coverage_size(rule, x)
                self.coverage_sizes[rule][i] = coverage

        size = sum(self.coverage_sizes[rule].values())

        return size

    def binary_fidelity(self, unit, x, y, ids=None, default=nan):
        """Evaluate the goodness of unit.
        Args:
            unit (Unit): The unit to evaluate.
            x (array): The data.
            y (array): The labels.
            ids (array): IDS of the given `patterns`, used to speed up evaluation.
            default (int): Default prediction for records not covered by the unit.
        Returns:
              (float): The unit's fidelity
        """
        if y is None:
            y = self.oracle.predict(x).squeeze()

        self.binary_fidelities[unit] = self.binary_fidelities.get(unit, binary_fidelity(unit, x, y, self, ids,
                                                                                        default=default))

        return self.binary_fidelities[unit]

    def binary_fidelity_model(self, units, scores, x, y, x_vl, y_vl,
                              evaluator=None, default=None, ids=None, ids_vl=None, k=5):
        """Evaluate the goodness of unit.
        Args:
            units (list): The units to evaluate.
            scores (iterable): Scores to use to select the top-k units.
            x (array): The data.
            y (array): The labels.
            x_vl (array): The validation data.
            y_vl (array): The validation labels.
            evaluator (Evaluator): Optional evaluator to speed-up computation.
            default (int): Default prediction for records not covered by the unit.
            ids (array): Unique identifiers to tell each element in @c apart.
            ids_vl (array): Unique identifiers to tell each element in @x_vl apart.
            k (int): Top-k rules to take.
        Returns:
              (float): The units fidelity.
        """
        if y is None:
            y = self.oracle.predict(x_vl).squeeze().round()

        coverage = self.coverage(units, x)

        predictions = []
        for record in range(len(x_vl)):
            record_coverage = argwhere(coverage[:, record]).ravel()
            if len(record_coverage) == 0:
                prediction = default
            else:
                companions_0 = [i for i in record_coverage if units[i].consequence == 0]
                companions_1 = [i for i in record_coverage if units[i].consequence == 1]
                scores_0 = scores[companions_0]
                scores_1 = scores[companions_1]
                argsort_scores_0 = flip(argsort(scores[companions_0])[-k:])
                argsort_scores_1 = flip(argsort(scores[companions_1])[-k:])
                top_scores_0 = scores_0[argsort_scores_0]
                top_scores_1 = scores_1[argsort_scores_1]

                if len(top_scores_0) == 0 and len(top_scores_1) > 0:
                    prediction = 1
                elif len(top_scores_1) == 0 and len(top_scores_0) > 0:
                    prediction = 0
                elif len(top_scores_1) == 0 and len(top_scores_0) == 0:
                    prediction = default
                else:
                    prediction = 0 if mean(top_scores_0) > mean(top_scores_1) else 1

            predictions.append(prediction)
        predictions = array(predictions)
        fidelity = 1 - hamming(predictions, y) if len(y) > 0 else 0

        return fidelity

    def score(self, unit, data, target, ids=None):
        """Score @unit against @data with labels @target.
        Args:
            unit (Unit): Unit to score.
            data (array): Data.
            target (array): Targets.
            ids (array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            (float): Score.
        """
        if ids is None:
            ids = array([str(i) for i in range(data.shape[0])])
        if unit not in self.scores:
            if self.oracle is not None or target is None:
                target = self.oracle.predict(data).squeeze()

            self.scores[unit] = self.binary_fidelity(unit, data, target, ids)

        return self.scores[unit]


########################
# Framework validation #
########################
def validate(rules, scores, oracle, vl, scoring='rrs', evaluator=None, alpha=0, beta=0, gamma=-1, max_len=inf):
    """Validate the given rules in the given ruleset.
    Arguments:
        rules (iterable): Iterable of rules to validate.
        scores(iterable): The scores.
        oracle (Predictor): Oracle to validate against.
        vl (ndarray): Validation set.
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
        return mean([len(r) for r in rules])

    def std_len(rules):
        return std([len(r) for r in rules])

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
    score_alpha_argprune = array([i for i in range(len(rules)) if scores[i] >= alpha and len(rules[i]) <= max_len])
    score_alpha_argprune = score_alpha_argprune if score_alpha_argprune.size <= gamma else score_alpha_argprune[:gamma]
    scoring_alpha_rules = [rules[i] for i in score_alpha_argprune]

    scoring_percentile = percentile(scores, beta) if beta >= 0 else median(scores)
    scoring_beta_argprune = array([i for i in range(len(rules))
                                    if scores[i] >= scoring_percentile and len(rules[i]) <= max_len])
    scoring_beta_argprune = scoring_beta_argprune if scoring_beta_argprune.size <= gamma\
                                                    else scoring_beta_argprune[:gamma]
    scoring_beta_rules = [rules[i] for i in scoring_beta_argprune]
    scoring_beta_rules_nr = len(scoring_beta_rules)

    validation = dict()
    validation['alpha']     = alpha
    validation['beta']      = beta
    validation['gamma']     = gamma
    validation['max_len']   = max_len
    validation['scoring']   = scoring

    scoring_fidelities = evaluator.binary_fidelity_model(scoring_alpha_rules, scores=scores,
                                                                x=x, y=y, x_vl=x, y_vl=y,
                                                                evaluator=evaluator,
                                                                default=majority_label)
    beta_scoring_fidelities = evaluator.binary_fidelity_model(scoring_beta_rules,
                                                               scores=scores[scoring_beta_argprune],
                                                               x=x, y=y, x_vl=x, y_vl=y,
                                                               evaluator=evaluator,
                                                               default=majority_label)

    validation['scoring-fidelities'] = scoring_fidelities
    validation['beta-scoring-fidelity'] = beta_scoring_fidelities
    validation['beta-scoring-rule_nr'] = scoring_beta_rules_nr
    validation['mean_length'] = mean_len(rules)
    validation['std_length'] = std_len(rules)

    validation['coverage'] = coverage_pct(rules, x)
    validation['beta-scoring-coverage'] = coverage_pct(scoring_beta_rules, x)
    validation['mean-beta-scoring-length'] = mean_len(scoring_beta_rules)
    validation['std-beta-scoring-length'] = std_len(scoring_beta_rules)

    # Predictions
    validation['mean_prediction'] = std([r.consequence for r in rules])
    validation['std-scoring-prediction'] = std([r.consequence for r in scoring_alpha_rules])
    validation['rule_reduction-alpha-scoring'] = len(scoring_alpha_rules) / len(rules)
    validation['rule_reduction-beta-scoring'] = len(scoring_beta_rules) / len(rules)
    validation['len_reduction-beta-scoring'] = len_reduction(validation['mean-beta-scoring-length'],
                                                             validation['mean_length'])
    validation['simplicity-beta-scoring'] = simplicity(validation['rule_reduction-beta-scoring'],
                                                       validation['len_reduction-beta-scoring'])
    features = rules[0].names
    validation['feature_frequency']                 = features_frequencies(rules, features)
    validation['scoring-beta-feature_frequency']    = features_frequencies(scoring_beta_rules, features)

    validation['scoring-beta-escore'] = harmonic_mean([validation['beta-scoring-fidelity'],
                                                        validation['simplicity-beta-scoring']])

    return validation
