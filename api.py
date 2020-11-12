"""API module."""
import os
import time
import pickle
import json

# Shut up, tensorflow!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import click
import logzero
from numpy import genfromtxt, inf, hstack, float as np_float
from tensorflow.keras.models import load_model

from logzero import logger

from models import Rule
from scoring import Scorer
from evaluators import MemEvaluator, validate


def pretty_print_results(results, dump_file):
    logger.info('-------------------------------------------------------')
    logger.info('Results stored in ' + dump_file)
    logger.info('alpha: '                                               + str(results['alpha']))
    logger.info('beta: '                                                + str(results['beta']))
    logger.info('gamma: '                                               + str(results['gamma']))
    logger.info('max_len: '                                             + str(results['max_len']))
    logger.info('score function: '                                      + str(results['scoring']))
    logger.info('[ALPHA-pruned ruleset] fidelity: '                     + str(results['scoring-fidelities']))
    logger.info('[BETA-pruned ruleset] fidelity: '                      + str(results['beta-scoring-fidelity']))
    logger.info('[BETA-pruned ruleset] #rules: '                        + str(results['beta-scoring-rule_nr']))
    logger.info('[BETA-pruned ruleset] coverage: '                      + str(100 * results['beta-scoring-coverage']) + '%')
    logger.info('[BETA-pruned ruleset] mean rule length: '              + str(results['mean-beta-scoring-length']))
    logger.info('[BETA-pruned ruleset] std rule length: '               + str(results['std-beta-scoring-length']))
    logger.info('[BETA-pruned ruleset] #rules reduction:'               + str(100 * results['rule_reduction-beta-scoring']) + '%')
    logger.info('[BETA-pruned ruleset] mean length reduction:'          + str(100 * results['len_reduction-beta-scoring']) + '%')
    logger.info('[BETA-pruned ruleset] explainability score:'           + str(results['simplicity-beta-scoring']))


@click.command()
@click.argument('rules',            type=click.Path(exists=True))
@click.argument('tr',               type=click.Path(exists=True))
@click.option('-vl',                default=None,                   help='Validation set, if any.')
@click.option('-o', '--oracle',     default=None,                   help='Black box to predict the dataset labels, if'
                                                                         ' any. Otherwise use the dataset labels.')
@click.option('-m', '--name',       default=None,                   help='Name of the log files.')
@click.option('-s', '--score',      default='rrs',                  help='Scoring function to use.'
                                                                        'Available functions are \'rrs\''
                                                                        ' (which includes fidelity scoring)'
                                                                        ' and \'coverage\'.'
                                                                        'Defaults to rrs.')
@click.option('-p', '--wp',         default=1.,                     help='Coverage weight. Defaults to 1.')
@click.option('-p', '--ws',         default=1.,                     help='Sparsity weight. Defaults to 1.')
@click.option('-a', '--alpha',      default=0.,                     help='Score pruning in [0, 1]. '
                                                                         'Defaults to 0 (no pruning).')
@click.option('-b', '--beta',       default=0.,                     help='Prune the rules under the beta-th percentile.'
                                                                        'Defaults to 0 (no pruning).')
@click.option('-g', '--gamma',      default=-1.,                    help='Maximum number of rules (>0) to use. '
                                                                        'Defaults to -1 use all.')
@click.option('-l', '--max_len',    default=-1.,                    help='Length pruning in [0, inf]. '
                                                                         'Defaults to -1 (no pruning).')
@click.option('-d', '--debug',      default=20,                     help='Debug level.')
def cl_run(rules, tr, vl, oracle, name, score='rrs', wp=1., ws=1., alpha=0, beta=0, gamma=-1, max_len=-1, debug=20):
    run(rules, score, name, oracle, tr, vl, coverage_weight=wp, sparsity_weight=ws, alpha=alpha, beta=beta, gamma=gamma,
        max_len=max_len, debug=debug)


def run(rules, scoring='rrs', name=None, oracle=None, tr=None, vl=None, coverage_weight=1., sparsity_weight=1.,
        alpha=0, beta=0, gamma=-1, max_len=-1, debug=20):
    """Run the Scoring framework on a set of rules.
    Arguments:
        rules (str): JSON file with the train set.
        scoring (str): Type of scoring to perform. Available names are 'rrs', which includes fidelity scoring, and
                        'coverage'. Defaults to 'rrs'.
        oracle (Union(str, None)): Oracle to score against.
        tr (str): Training set.
        vl (str): Validation set, if any.
        name (str): Name for the output logs.
        coverage_weight (float): Coverage weight vector.
        sparsity_weight (float): Sparsity weight vector.
        alpha (float): Pruning hyperparameter, rules with score less than `alpha` are removed from the result.
        beta (float): Pruning hyperparameter, rules with score less than the `beta`-percentile are removed from the
                        result.
        gamma (int): Maximum number of rules to use.
        max_len (int): Pruning hyperparameter, rules with length more than `max_len` are removed from the result.
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
        name = tr + str(time.time())

    # Info LOG
    logger.info('Rules: '           + str(rules))
    logger.info('name: '            + str(name))
    logger.info('score: '           + str(scoring))
    logger.info('oracle: '          + str(oracle))
    logger.info('vl: '              + str(vl))
    logger.info('coverage weight: ' + str(coverage_weight))
    logger.info('sparsity weight: ' + str(sparsity_weight))
    logger.info('alpha: '           + str(alpha))
    logger.info('beta: '            + str(beta))
    logger.info('max len: '         + str(max_len))

    logger.info('Loading validation... ')
    data = genfromtxt(tr, delimiter=',', names=True)
    names = data.dtype.names
    training_set = data.view(np_float).reshape(data.shape + (-1,))
    if vl is not None:
        data = genfromtxt(vl, delimiter=',', names=True)
        validation_set = data.view(np_float).reshape(data.shape + (-1,))
    else:
        validation_set = None

    # Run Scoring
    logger.info('Loading ruleset...')
    rules = Rule.from_json(rules, names)
    rules = [r for r in rules if len(r) > 0]

    logger.info('Loading oracle...')
    if oracle is not None:
        if oracle.endswith('.h5'):
            oracle = load_model(oracle)
        elif oracle.endswith('.pickle'):
            with open(oracle, 'rb') as log:
                oracle = pickle.load(log)
        else:
            return
        if validation_set is not None:
            validation_set = hstack((validation_set[:, :-1],
                                     oracle.predict(validation_set[:, :-1].round()).reshape((validation_set.shape[0], 1))))
        training_set = hstack((training_set[:, :-1],
                               oracle.predict(training_set[:, :-1].round()).reshape((training_set.shape[0], 1))))

    logger.info('Creating scorer...')
    evaluator = MemEvaluator(oracle=oracle)
    scorer = Scorer(score=scoring, evaluator=evaluator, oracle=oracle)

    logger.info('Scoring...')
    scores = scorer.score(rules, training_set, coverage_weight, sparsity_weight)

    logger.info('Storing scores...')
    storage_file = name + '.csv'
    scorer.save(scores=scores, path=storage_file)

    # Validation
    logger.info('Validating...')
    validation_dic = validate(rules, scores, oracle=oracle,
                              vl=validation_set if validation_set is not None else training_set,
                              scoring=scoring,
                              alpha=alpha, beta=beta,
                              gamma=len(rules) if gamma < 0 else int(gamma),
                              max_len=inf if max_len <= 0 else max_len)
    validation_dic['coverage'] = coverage_weight
    validation_dic['sparsity'] = sparsity_weight
    validation_dic['scoring'] = scoring

    # Store info on JSON
    out_str = name + '.results.json'
    if os.path.isfile(out_str):
        with open(out_str, 'r') as log:
            out_dic = json.load(log)
        out_dic['runs'].append({
                'scoring': scoring,
                'coverage': coverage_weight,
                'sparsity': sparsity_weight,
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
                'coverage': coverage_weight,
                'sparsity': sparsity_weight,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'max_len': max_len,
                'results': validation_dic
            }]
        }
    with open(out_str, 'w') as log:
        json.dump(out_dic, log)

    pretty_print_results(validation_dic, out_str)


if __name__ == '__main__':
    cl_run()
