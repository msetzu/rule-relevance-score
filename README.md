# RRS ~ Rule Relevance Score

## Development
This git repository follows the [Git Flow](https://jeffkreeftmeijer.com/git-flow/) development model.
The `master` branch holds the stable release version, while the `develop` branch holds the development version.
Additional `feature/$feature_name` branches departing from `develop` are used to develop each additional feature,
and are then merged in `develop`.

## Repository organization
Now, on the file and directory structure. 

- `docs` holds the `html` API documentation. To start a simple http server run
```
python3 -m http.server
``` 
which will open an http server on `0.0.0.0:8000` from which you can browse the documentation.
- `data` holds the datasets and rules used in the experimentation.
    - `$dataset.h5`, DNN for dataset `$dataset`;
    - `$dataset.rf.pickle`, RF for dataset `$dataset`;
    - `$dataset.discretization.json`, bin data for `corels` and `sbrl` encodings;
    - `$dataset.dnn.json`, DNN rules for dataset `$dataset`;
    - `$dataset.rf.json`, RF rules for dataset `$dataset`;
    - `$dataset.sbrl.dnn.json`, SBRL rules for dataset `$dataset`, DNN black box;
    - `$dataset.sbrl.rf.json`, SBRL rules for dataset `$dataset`, RF black box;
    - `$dataset.corels.dnn.json`, CORELS rules for dataset `$dataset`, DNN black box;
    - `$dataset.corels.rf.json`, CORELS rules for dataset `$dataset`, RF black box;
    - `$dataset.cpar.dnn.json`, CPAR rules for dataset `$dataset`, DNN black box;
    - `$dataset.cpar.rf.json`, CPAR rules for dataset `$dataset`, RF black box;
- `experiments` contains the experiments' output (`output` folder), auxiliary modules and notebooks (`notebooks` folder)
    to reproduce the experiments     
- `requirements.txt` holds the `python3` dependencies to run the project.

## Run code
In order to run the code you need to first install the dependencies:
```
pip3 install -r requirements.txt
```

You can change the tests verbosity by providing the one of the logging parameters of decreasing verbosity: `10, 20, 30, 40, 50`.
Defaults to `10`.

You can run `RRS` from the command line with the following syntax:
```
python3 api.py $rules $TR $TS
```
where `$rules` is the rules file, and `$TR` and `$TS` are the training and validation files.
You can customize the run with the following options:
- `-o/--oracle` path to the black box to use, if any.
- `-m/--name` base name for the log files
- `-s/--score` type of scoring function to use. `r2`, `coverage`, `fidelity` are valid functions.
- `-p/--wp` coverage weight. Defaults to `1`.
- `-p/--ws` sparsity weight. Defaults to `1`.
- `-a/--alpha` hard pruning threshold. Defaults to `0`.
- `-b/--beta` percentile pruning threshold. Defaults to `0`.
- `-g/--gamma` pruning factor. Keep at most `gamma` rules.
- `-l/--max_len` pruning factor. Keep rules shorter than `max_len`.
- `-k/--k` best rules to query to predict.
- `--debug=$d` to set the logging level

For instance, you can run
```
python3 api.py data/compas.dnn.json data/compas_tr.csv data/compas_ts.csv \
                -o data/compas.h5 --name=compast --score r2 --beta 75
```
You can find the whole documentation by running `--help` on the running script:
```
$ python3 api.py --help
Usage: api.py [OPTIONS] RULES TR VL

Options:
  -o, --oracle TEXT
  -m, --name TEXT      Name of the log files.
  -s, --score TEXT     Scoring function to use.Available functions are 'r2'
                       (which includes fidelity scoring) and
                       'coverage'.Defaults to r2.
  -p, --wp FLOAT       Coverage weight. Defaults to 1.
  -p, --ws FLOAT       Sparsity weight. Defaults to 1.
  -a, --alpha FLOAT    Score pruning in [0, 1]. Defaults to 0 (no pruning).
  -b, --beta FLOAT     Score percentile pruning in [0, 1]. Defaults to 0 (no
                       pruning).
  -g, --gamma FLOAT    Maximum number of rules (>0) to use. Defaults to -1 use
                       all.
  -l, --max_len FLOAT  Length pruning in [0, inf]. Defaults to -1 (no
                       pruning).
  -k, --k INTEGER      Number of rules to use at prediction time. Defaults to
                       1.
  -d, --debug INTEGER  Debug level.
  --help               Show this message and exit.


```