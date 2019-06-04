"""Scoring measures."""
from eval import MemEvaluator

from scipy.spatial.distance import cdist
from numpy import array, isnan, isinf, vectorize, sum, matmul, savetxt, hstack


class Scorer:
    """Score the provided rules."""

    def __init__(self, score='r2', evaluator=None, oracle=None, **kwargs):
        """Scorer with hyperparameters given by kwargs.

        Arguments:
            score (str): Type of scoring to perform. Available
                        names are 'r2', which includes fidelity scoring,
                        and 'coverage'. Defaults to 'r2'.
            evaluator (Evaluator): Evaluator to speed up computation.
            oracle (Predictor): Oracle against which to score.
            kwargs: Keyword arguments for the scoring hyperparameters.
                    The following keys are accepted:
                    w_p (float): Coverage weight.
                    w_s (float): Sparsity weight.
        """
        self.scoring = score
        self.evaluator = evaluator if evaluator is not None else MemEvaluator(oracle)
        self.oracle = oracle
        self.w_p = kwargs.get('w_p', 1.)
        self.w_s = kwargs.get('w_s', 1.)

    def local_score(self, rules, patterns, w_p, w_s, pattern_distance='euclidean'):
        """Score the provided @rules against @patterns.
        Arguments:
            rules (list): A list of rules to score.
            patterns (ndarray): The pattern against which to score
                                        the @rules.
            w_p (float): Coverage weight.
            w_s (float): Sparsity weight.
            pattern_distance (str): Distance measure, defaults to
                                    euclidean distance, but can be changed
                                    with the str @distance parameter.
                                    Any scipy distance is supported.

        Returns:
            (ndarray): A #rules ndarray vector where
                        the i^th entry is the score for the i^th input rule.
        """
        p, r = patterns.shape[0], len(rules)
        distance_matrix = cdist(patterns, patterns, metric=pattern_distance)

        # Coverage
        coverage_matrix = self.evaluator.coverage(rules, patterns[:, :-1])
        coverage_vector = coverage_matrix.sum(axis=1) / p

        # Sparsity
        sparsity_vector = sum(matmul(coverage_matrix, distance_matrix), axis=1)
        max_sparsity = sparsity_vector.max()
        sparsity_vector = sparsity_vector / max_sparsity if max_sparsity > 0 else sparsity_vector

        # Scores
        pattern_coverage_score = 1 - sum(coverage_matrix, axis=0) / r
        rule_coverage_score = matmul(coverage_matrix, pattern_coverage_score)
        rule_coverage_score = rule_coverage_score / rule_coverage_score.max()
        rule_coverage_score = vectorize(lambda x: 0 if isnan(x) else x)(rule_coverage_score)

        scores = (w_p * coverage_vector + w_s * sparsity_vector + rule_coverage_score) / (1 + w_p + w_s)

        return scores

    def global_score(self, rules, local_scores, data):
        """Score global rules with local scores against x.

        Arguments:
            rules (list): The rules.
            local_scores (array): Local scores.
            data (ndarray): Dataset against which to score the rules.

        Returns:
        (ndarray): An array of scores.
        """
        p, _ = data.shape
        x, y = data[:, :-1], data[:, -1]
        y = self.oracle.predict(data[:, :-1]).squeeze() if self.oracle is not None else y
        majority_label = round(y.sum() / len(y))
        # Coverage component
        global_coverage = self.evaluator.coverage(rules, x)
        global_pure_coverage = self.evaluator.coverage(rules, x, y)
        pure_coverage_vector = array([self.evaluator.binary_fidelity(rule, x, y, ids=None, default=majority_label)
                                        for rule in rules])

        # Friendliness component
        pattern_frequencies = abs(sum(global_coverage, axis=0))
        friendliness_scores = (1 / pattern_frequencies).squeeze()
        friendliness_scores[isinf(friendliness_scores)] = 0
        friendliness_vector = matmul(global_pure_coverage, friendliness_scores) / p

        global_scores = (pure_coverage_vector + friendliness_vector + local_scores) / 3

        return global_scores

    def score(self, rules, patterns, w_p, w_s, pattern_distance='euclidean'):
        """Score the provided @rules against @patterns.
        Arguments:
            rules (list): A list of rules to score.
            patterns (ndarray): The pattern against which to score
                                        the @rules.
            w_p (float): Coverage weight.
            w_s (float): Sparsity weight.
            pattern_distance (str): Distance measure, defaults to
                                    euclidean distance, but can be changed
                                    with the str @distance parameter.
                                    Any scipy distance is supported.

        Returns:
            (ndarray): A #rules ndarray vector where
                        the i^th entry is the score for the i^th input rule.
        """
        x = patterns[:, :-1]
        y = self.oracle.predict(patterns[:, :-1]).round().squeeze() if self.oracle is not None else patterns[:, -1]
        majority_label = round(y.sum() / len(y))
        data = hstack((x, y.reshape(-1, 1)))

        if self.scoring == 'r2':
            local_scores = self.local_score(rules, data, w_p, w_s, pattern_distance)
            global_scores = self.global_score(rules, local_scores, data)
        elif self.scoring == 'fidelity':
            global_scores = array([self.evaluator.binary_fidelity(rule, x, y, ids=None, default=majority_label)
                                   for rule in rules])
        elif self.scoring == 'coverage':
            global_scores = self.evaluator.coverage(rules, x).sum(axis=1) / x.shape[0]
        else:
            global_scores = None

        return global_scores

    def save(self, scores, path):
        """Save this scorer's `scores` in `path`.

        Args:
            scores (ndarray): Scores.
            path (str): Storage location of the csv scores.
                        Storage file is `path`.csv
        """
        with open(path, 'w') as log:
            savetxt(log, scores, newline=',')