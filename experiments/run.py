"""Run experiments."""
import sys
import os
from itertools import product

from tqdm import tqdm

wd = os.getcwd() + '/'
data_folder = wd + '../data/'
sys.path.append(wd + '../')


import api


def main(debug=20):
    """Run."""
    datasets = ['adult', 'german', 'compas', 'churn']

    rule_files = [data_folder + dataset + '.json' for dataset in datasets]
    oracle_files = [data_folder + dataset + '.h5' for dataset in datasets]
    tr_data_files = [data_folder + dataset + '_tr.csv' for dataset in datasets]
    ts_data_files = [data_folder + dataset + '_ts.csv' for dataset in datasets]
    hyperparameters_space = list(product([25, 50, 75, 90, 100], [25, 50, 75, 90, 100]))

    for dataset, rules, oracle, tr, ts in zip(datasets, rule_files, oracle_files, tr_data_files, ts_data_files):
        for wp, ws in tqdm(hyperparameters_space):
            api.run(rules, oracle=None, vl=tr, name=dataset, wp=wp / 10, ws=ws / 10, debug=debug)
            api.run(rules, oracle=None, vl=ts, name=dataset, wp=wp / 10, ws=ws / 10, debug=debug)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
