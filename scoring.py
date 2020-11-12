"""Scoring measures."""
from evaluators import MemEvaluator

from scipy.spatial.distance import cdist
import numpy as np


class Scorer:
    """Score the provided rules."""

    def __init__(self, score='rrs', evaluator=None, oracle=None):
        """Scorer with hyperparameters given by kwargs.

        Arguments:
            score (str): Type of scoring to perform. Available
                        names are 'rrs', which includes fidelity scoring,
                        and 'coverage'. Defaults to 'rrs'.
            evaluator (Evaluator): Evaluator to speed up computation.
            oracle (Predictor): Oracle against which to score.
        """
        self.scoring = score
        self.evaluator = evaluator if evaluator is not None else MemEvaluator(oracle)
        self.oracle = oracle

    def local_score(self, rules, data, coverage_weight, sparsity_weight, distance='euclidean'):
        """Score the provided @rules against @patterns.
        Arguments:
            rules (list): A list of rules to score.
            data (numpy.ndarray): The pattern against which to score the `rules`.
            coverage_weight (float): Coverage weight.
            sparsity_weight (float): Sparsity weight.
            distance (Union(str, function)): Distance measure, defaults to euclidean distance, but can be
                                                    changed to any scipy distance. Can also be function with the
                                                    following signature: `f(x:numpy.ndarray) -> numpy.ndarray`
                                                    that, given a numpy.ndarray matrix of instances returns an
                                                    n-by-n matrix where entry i,j holds the distance between
                                                    instance i and instance j. Diagonal entries set to 0.
        Returns:
            numpy.ndarray: A #rules numpy.ndarray vector where
                        the i^th entry is the score for the i^th input rule.
        """
        p, r = data.shape[0], len(rules)
        if isinstance(distance, str):
            distance_matrix = cdist(data, data, metric=distance)
        else:
            distance_matrix = distance(data)

        # Coverage
        coverage_matrix = self.evaluator.coverage(rules, data[:, :-1])
        coverage_vector = coverage_matrix.sum(axis=1) / p

        # Sparsity
        sparsity_vector = np.sum(np.matmul(coverage_matrix, distance_matrix), axis=1)
        max_sparsity = sparsity_vector.max()
        sparsity_vector = sparsity_vector / max_sparsity if max_sparsity > 0 else sparsity_vector

        # Scores
        pattern_coverage_score = 1 - np.sum(coverage_matrix, axis=0) / r
        rule_coverage_score = np.matmul(coverage_matrix, pattern_coverage_score)
        rule_coverage_score = rule_coverage_score / rule_coverage_score.max()
        rule_coverage_score = np.vectorize(lambda x: 0 if np.isnan(x) else x)(rule_coverage_score)

        scores = (coverage_weight * coverage_vector + sparsity_weight * sparsity_vector +
                  rule_coverage_score) / (1 + coverage_weight + sparsity_weight)

        return scores

    def global_score(self, rules, local_scores, data):
        """Score global rules with local scores against x.

        Arguments:
            rules (list): The rules.
            local_scores (array): Local scores.
            data (numpy.ndarray): Dataset against which to score the rules.

        Returns:
        (numpy.ndarray): An array of scores.
        """
        p, _ = data.shape
        x, y = data[:, :-1], data[:, -1]
        y = self.oracle.predict(data[:, :-1]).squeeze() if self.oracle is not None else y
        majority_label = round(y.sum() / len(y))
        # Coverage component
        global_coverage = self.evaluator.coverage(rules, x)
        global_pure_coverage = self.evaluator.coverage(rules, x, y)
        pure_coverage_vector = np.array([self.evaluator.binary_fidelity(rule, x, y, ids=None, default=majority_label)
                                        for rule in rules])

        # Friendliness component
        pattern_frequencies = np.abs(np.sum(global_coverage, axis=0))
        friendliness_scores = (1 / pattern_frequencies).squeeze()
        friendliness_scores[np.isinf(friendliness_scores)] = 0
        friendliness_vector = np.matmul(global_pure_coverage, friendliness_scores) / p

        global_scores = (pure_coverage_vector + friendliness_vector + local_scores) / 3

        return global_scores

    def score(self, rules, data, coverage_score, sparsity_score, distance_function='euclidean'):
        """Score the provided `rules` against `patterns`.
        Arguments:
            rules (list): A list of rules to score.
            data (numpy.ndarray): The pattern against which to score
                                        the @rules.
            coverage_score (float): Coverage weight.
            sparsity_score (float): Sparsity weight.
            distance_function (str): Distance measure, defaults to
                                    euclidean distance, but can be changed
                                    with the str `distance` parameter.
                                    Any scipy distance is supported.

        Returns:
            (numpy.ndarray): A #rules numpy.ndarray vector where
                        the i^th entry is the score for the i^th input rule.
        """
        x = data[:, :-1]
        y = self.oracle.predict(data[:, :-1]).round().squeeze() if self.oracle is not None else data[:, -1]
        majority_label = round(y.sum() / len(y))
        data = np.hstack((x, y.reshape(-1, 1)))

        if self.scoring == 'rrs':
            local_scores = self.local_score(rules, data, coverage_score, sparsity_score, distance_function)
            global_scores = self.global_score(rules, local_scores, data)
        elif self.scoring == 'fidelity':
            global_scores = np.array([self.evaluator.binary_fidelity(rule, x, y, ids=None, default=majority_label)
                                   for rule in rules])
        elif self.scoring == 'coverage':
            global_scores = self.evaluator.coverage(rules, x).sum(axis=1) / x.shape[0]
        else:
            global_scores = None

        return global_scores

    @staticmethod
    def filter(rules, scores, alpha=0, beta=0, gamma=np.inf, max_len=np.inf):
        """
        Filter `rules` according to their score (`alpha`, `beta`), maximum length (`max_len`) and number (`gamma`).
        Args:
            rules (list): Rules to filter
            scores (np.ndarray): Rule's score
            alpha (float): Filter out rules with score under `alpha`
            beta (float): Filter out rules with score under the `beta`th-percentile
            gamma (float): Maximum number of rules to return
            max_len (float): Filter out rules longer than `max_len`

        Returns:
            list: Filtered rules
        """
        n = len(rules)
        beta_value = np.percentile(scores, beta)
        idx = [i for i in range(n) if scores[i] >= alpha and
               scores[i] >= beta_value and
               len(rules[i]) <= max_len][:gamma]
        filtered_rules = [rules[i] for i in idx]

        return filtered_rules



    def save(self, scores, path):
        """Save this scorer's `scores` in `path`.

        Args:
            scores (numpy.ndarray): Scores.
            path (str): Storage location of the csv scores.
                        Storage file is `path`.csv
        """
        with open(path, 'w') as log:
            np.savetxt(log, scores, newline=',')
