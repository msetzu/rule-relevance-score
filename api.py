"""API module."""
import os
import time
import pickle
import json

# Shut up, tensorflow!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import click
import logzero
from numpy import genfromtxt, argwhere, array, inf, hstack
import tensorflow as tf
from keras.models import load_model

from logzero import logger

from models import Rule
from measures import Scorer
from eval import MemEvaluator, validate

tf.logging.set_verbosity(tf.logging.ERROR)


class DummyOracle:
    """Dummy oracle, just repeats what has been given, majority vote otherwise."""

    def __init__(self, data):
        """Constructor."""
        self.x = data[:, :-1]
        self.y = data[:, -1]
        self.majority_vote = int(.5 + self.y.sum() / self.y.shape[0])

    def predict(self, x):
        """Predict x, nan if unknown at construction time."""
        predictions = list()
        for x_ in x:
            idx = argwhere((self.x == x_).all(axis=1))

            if len(idx) > 0:
                predictions.append(self.y[idx.item(0)])
            else:
                predictions.append(self.majority_vote)
        predictions = array(predictions)

        return predictions

@click.command()
@click.argument('rules',            type=click.Path(exists=True))
@click.argument('tr',               type=click.Path(exists=True))
@click.argument('vl',               type=click.Path(exists=True))
@click.option('-o', '--oracle',     default=None)
@click.option('-m', '--name',       default=None,                   help='Name of the log files.')
@click.option('-s', '--score',      default='r2',                   help='Scoring function to use.'
                                                                        'Available functions are \'r2\''
                                                                        ' (which includes fidelity scoring)'
                                                                        ' and \'coverage\'.'
                                                                        'Defaults to r2.')
@click.option('-p', '--wp',         default=1.,                     help='Coverage weight. Defaults to 1.')
@click.option('-p', '--ws',         default=1.,                     help='Sparsity weight. Defaults to 1.')
@click.option('-a', '--alpha',      default=0.,                     help='Score pruning in [0, 1]. '
                                                                     'Defaults to 0 (no pruning).')
@click.option('-b', '--beta',       default=0.,                     help='Score percentile pruning in [0, 1]. '
                                                                        'Defaults to 0 (no pruning).')
@click.option('-g', '--gamma',      default=-1.,                    help='Maximum number of rules (>0) to use. '
                                                                        'Defaults to -1 use all.')
@click.option('-l', '--max_len',    default=-1.,                    help='Length pruning in [0, inf]. '
                                                                         'Defaults to -1 (no pruning).')
@click.option('-k', '--k',          default=1,                      help='Number of rules to use at prediction time. '
                                                                         'Defaults to 1.')
@click.option('-d', '--debug',      default=20,                     help='Debug level.')
def cl_run(rules, oracle, tr, vl, name, score='r2', wp=1., ws=1., alpha=0, beta=0, gamma=-1, max_len=-1, k=1, debug=20):
    run(rules, score, name, oracle, tr, vl, wp=wp, ws=ws, alpha=alpha, beta=beta, gamma=gamma, max_len=max_len, k=k,
        debug=debug)


def run(rules, scoring='r2', name=None, oracle=None, tr=None, vl=None, wp=1., ws=1., alpha=0, beta=0, gamma=-1, max_len=-1, k=1, debug=20):
    """Run the Scoring framework on a set of rules.
    Arguments:
        rules (str): JSON file with the train set.
        scoring (str): Type of scoring to perform. Available
                        names are 'r2', which includes fidelity scoring,
                        and 'coverage'. Defaults to 'r2'.
        oracle (Union(str, None)): Oracle to score against.
        tr (str): Training set.
        vl (str): Validation set.
        name (str): Name for the output logs.
        wp (float): Coverage weight vector.
        ws (float): Sparsity weight vector.
        alpha (float): Pruning hyperparameter, rules with score
                        less than `alpha` are removed from the
                        result.
        beta (float): Pruning hyperparameter, rules with score
                        less than the `beta`-percentile are removed from the
                        result.
        gamma (int): Maximum number of rules to use.
        max_len (int): Pruning hyperparameter, rules with length
                        more than `max_len` are removed from the
                        result.
        debug (int): Minimum debug level.
    """
    # Set-up debug
    if debug == 10:
        min_log = logzero.logging.DEBUG
    elif debug == 20:
        min_log = logzero.logging.INFO
    elif debug == 30:
        min_log = logzero.logging.WARNING
    elif debug == 40:
        min_log = logzero.logging.ERROR
    elif debug == 50:
        min_log = logzero.logging.CRITICAL
    else:
        min_log = 0

    logzero.loglevel(min_log)

    if name is None:
        name = str(time.time())

    # Info LOG
    logger.info('Rules: '   + str(rules))
    logger.info('name: '    + str(name))
    logger.info('score: '   + str(scoring))
    logger.info('k: '       + str(k))
    logger.info('oracle: '  + str(oracle))
    logger.info('vl: '      + str(vl))
    logger.info('wp: '      + str(wp))
    logger.info('ws: '      + str(ws))
    logger.info('alpha: '   + str(alpha))
    logger.info('beta: '    + str(beta))
    logger.info('max len: ' + str(max_len))

    logger.info('Loading validation... ')
    tr_set = genfromtxt(tr, delimiter=',')
    validation_set = genfromtxt(vl, delimiter=',')

    # Run Scoring
    logger.info('Loading ruleset...')
    rules = Rule.from_json(rules)
    rules = [r for r in rules if len(r) > 0]

    logger.info('Loading oracle...')
    if oracle is not None:
        oracle_name = oracle.split('.')[0]
        if oracle.endswith('.h5'):
            oracle = load_model(oracle)
        elif oracle.endswith('.pickle'):
            with open(oracle, 'rb') as log:
                oracle = pickle.load(log)
        else:
            return
        validation_set = hstack((validation_set[:, :-1],
                                 oracle.predict(validation_set[:, :-1].round()).reshape((validation_set.shape[0], 1))))
        tr_set = hstack((tr_set[:, :-1],
                         oracle.predict(tr_set[:, :-1].round()).reshape((tr_set.shape[0], 1))))

    logger.info('Creating scorer...')
    evaluator = MemEvaluator(oracle=oracle)
    scorer = Scorer(score=scoring, evaluator=evaluator, oracle=oracle, wp=wp, ws=ws)

    logger.info('Scoring...')
    scores = scorer.score(rules, tr_set, wp, ws)

    logger.info('Storing scores...')
    storage_file = name + '.csv'
    scorer.save(scores=scores, path=storage_file)

    # Validation
    logger.info('Validating...')
    evaluator = MemEvaluator(oracle=oracle)
    validation_dic = validate(rules, scores, oracle, validation_set, evaluator, alpha=alpha, beta=beta,
                              gamma=len(rules) if gamma < 0 else int(gamma),
                              max_len=inf if max_len <= 0 else max_len, k=k, debug=debug)
    validation_dic['coverage'] = wp
    validation_dic['sparsity'] = ws
    validation_dic['scoring'] = scoring
    validation_dic['black_box'] = oracle_name if oracle is not None else ''

    # Store info on JSON
    out_str = name + '.results.json'
    if os.path.isfile(out_str):
        with open(out_str, 'r') as log:
            out_dic = json.load(log)
        out_dic['runs'].append({
                'scoring': scoring,
                'coverage': wp,
                'sparsity': ws,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'max_len': max_len,
                'results': validation_dic
            })
    else:
        out_dic = {
            'name': name,
            'runs': [{
                'scoring': scoring,
                'coverage': wp,
                'sparsity': ws,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'max_len': max_len,
                'results': validation_dic
            }]
        }
    with open(out_str, 'w') as log:
        json.dump(out_dic, log)
    logger.info('Done.')


if __name__ == '__main__':
    cl_run()
