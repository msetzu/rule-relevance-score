# RRS ~ Rule Relevance Score
Explanations come in two forms: local, explaining a single model prediction, and global, explaining all model predictions. The Local to Global (L2G) problem consists in bridging these two family of explanations. Simply put, we generate global explanations by merging local ones.

You can find a more detailed explanation in the [conference poster](https://drive.google.com/file/d/1YlYNMG0eUmWR3loFOOX3OQ4PMct0gdyr/view?usp=sharing).

## The algorithm

Local and global explanations are provided in the form of decision rules:

```
age < 40, income > 50000, status = married, job = CEO ⇒ grant loan application
```

This rule describes the rationale followed given by an unexplainable model to grant the loan application to an individual younger than 40 years-old, with an income above 50000$, married and currently working as a CEO.

---

## Setup

```bash
git clone https://github.com/msetzu/rule-relevance-score/
cd rule-relevance-score
```

Dependencies are listed in `requirements.txt`, a virtual environment is advised:

```bash
mkvirtualenv rule-relevance-score # optional but recommended
pip3 install -r requirements.txt
```

## Running the code

### Python interface
```python
from tensorflow.keras.models import load_model
from numpy import genfromtxt, float as np_float
import logzero

from scoring import Scorer
from models import Rule

# Set log profile: INFO for normal logging, DEBUG for verbosity
logzero.loglevel(logzero.logging.INFO)

# Load black box: optional! Use black_box = None to use the dataset labels
black_box = load_model('data/dummy/dummy_model.h5')
# Load data and header
data = genfromtxt('data/dummy/dummy_dataset.csv', delimiter=',', names=True)
features_names = data.dtype.names
tr_set = data.view(np_float).reshape(data.shape + (-1,))

# Load local explanations
local_explanations = Rule.from_json('data/dummy/dummy_rules.json', names=features_names)

# Create a RRS instance for `black_box`
scorer = Scorer('rrs', oracle=black_box)
# Fit the model
coverage_weight = 1.
sparsity_weight = 1.
scores = scorer.score(local_explanations, tr_set,
                        coverage_weight=coverage_weight,
                        sparsity_weight=sparsity_weight)
```
You can filter rules with the static method `scoring.Scorer.filter():`
```python
# filter by score percentile
filtered_rules = Scorer.filter(rules, scores, beta=beta)
# filter by score
filtered_rules = Scorer.filter(rules, scores, alpha=alpha)
# filter by maximum length
filtered_rules = Scorer.filter(rules, scores, max_len=max_len)
# filter by maximum number
filtered_rules = Scorer.filter(rules, scores, gamma=gamma)
# filter by any combination
filtered_rules = Scorer.filter(rules, scores, beta=beta, gamma=gamma)
filtered_rules = Scorer.filter(rules, scores, alpha=alpha, max_len=max_len)
```
You can directly validate the model with the built-in function `validate`:
```python
from evaluators import validate
validation_dic = validate(rules, scores, oracle=oracle,
                            vl=tr_set,
                            scoring='rrs',
                            alpha=0, beta=beta,
                            gamma=len(rules) if gamma < 0 else int(gamma),
                            max_len=inf if max_len <= 0 else max_len)
```

## Command line interface
You can run `RRS` from the command line with the following syntax:
```
python3 api.py $rules $TR $TS --oracle $ORACLE --name $NAME
```
where `$rules` is the rules file; `$TR` and `$TS` are the training and (optionally) validation sets; `$ORACLE` is the path to your black box (in hdf5 format) and `$NAME` is the name of your desired output file containing the JSON validation dictionary.
If you are interested to check formatting, the folder `data/dummy/` contains a dummy input example.
You can customize the run with the following options:
- `-o/--oracle` path to the black box to use, if any.
- `-m/--name` base name for the log files
- `-s/--score` type of scoring function to use. `rrs`, `coverage`, `fidelity` are valid functions.
- `-p/--wp` coverage weight. Defaults to `1`.
- `-p/--ws` sparsity weight. Defaults to `1`.
- `-a/--alpha` hard pruning threshold. Defaults to `0`.
- `-b/--beta` percentile pruning threshold. Defaults to `0`.
- `-g/--gamma` pruning factor. Keep at most `gamma` rules.
- `-l/--max_len` pruning factor. Keep rules shorter than `max_len`.
- `--debug=$d` to set the logging level

This documentation is available by running `--help` on the running script:
```shell script
$ python3 api.py --help
Usage: api.py [OPTIONS] RULES TR

Options:
  -vl TEXT             Validation set, if any.
  -o, --oracle TEXT    Black box to predict the dataset labels, if any.
                       Otherwise use the dataset labels.

  -m, --name TEXT      Name of the log files.
  -s, --score TEXT     Scoring function to use.Available functions are 'rrs'
                       (which includes fidelity scoring) and
                       'coverage'.Defaults to rrs.

  -p, --wp FLOAT       Coverage weight. Defaults to 1.
  -p, --ws FLOAT       Sparsity weight. Defaults to 1.
  -a, --alpha FLOAT    Score pruning in [0, 1]. Defaults to 0 (no pruning).
  -b, --beta FLOAT     Prune the rules under the beta-th percentile.Defaults
                       to 0 (no pruning).

  -g, --gamma FLOAT    Maximum number of rules (>0) to use. Defaults to -1 use
                       all.

  -l, --max_len FLOAT  Length pruning in [0, inf]. Defaults to -1 (no
                       pruning).

  -d, --debug INTEGER  Debug level.
  --help               Show this message and exit.
```
## Run on your own dataset

RRS has a strict format on input data. It accepts tabular datasets and binary classification tasks. You can find a dummy example for each of these formats in `/data/dummy/`.

#### Rules [`/data/dummy/dummy_rules.json`]

Local rules are to be stored in a `JSON` format:

```json
[
    {"22": [30.0, 91.9], "23": [-Infinity, 553.3], "label": 0},
    ...
]
```

Each rule in the list is a dictionary with an arbitrary (greater than 2) premises. The rule prediction ({0, 1}) is stored in the key `label`. Premises on features are stored according to their ordering and bounds: in the above, `"22": [-Infinity, 91.9]` indicates the premise "feature number 22 has value between 30.0 and 91.9".

#### Black boxes [`/data/dummy/dummy_model.h5`]

Black boxes (if used) are to be stored in a `hdf5` format if given through command line. If given programmatically instead, it suffices that they implement the `Predictor` interface:

```python
class Predictor:
    @abstractmethod
    def predict(self, x):
        pass
```

when called to predict `numpy.ndarray:x` the predictor shall return its predictions in a `numpy.ndarray` of integers.

#### Training data[`/data/dummy/dummy_dataset.csv`]

Training data is to be stored in a csv, comma-separated format with features names as header. The classification labels should have feature name `y`.

---
## Docs

You can find the software documentation in the `/html/` folder and a conference poster on RRS can be found [here](https://drive.google.com/file/d/1YlYNMG0eUmWR3loFOOX3OQ4PMct0gdyr/view?usp=sharing).

The work is has been published as a conference poster at [XKDD 2019](https://kdd.isti.cnr.it/xkdd2019/) at the joint [ECML/PKDD 2019](https://ecmlpkdd2019.org/) conference.

