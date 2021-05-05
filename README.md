# Evaluation of NumPyro autoguides on Stan programs

This project evaluates [NumPyro autoguides](http://num.pyro.ai/en/stable/autoguide.html) for variational inference on Stan models using [PosteriorDB](https://github.com/stan-dev/posteriordb).
It relies on the fork of the Stanc3 compiler available at https://github.com/deepppl/stanc3 to compile the Stan programs to NumPyro.

## Getting Started

You need to install the following dependencies:
- [opam](https://opam.ocaml.org/): the OCaml package manager
- [bazel](https://bazel.build/): required by tensorflow-probability

Stanc requires version 4.07.0 of OCaml which can be installed with:
```
opam switch create 4.07.0
opam switch 4.07.0
```

Then simply run the following command to install all the dependencies, including the compiler.
```
make init
```

## Experiments

The complete experiment consists of the execution of all the models of [PosteriorDB](https://github.com/stan-dev/posteriordb) that have a reference result with all the autoguides provided by Stan and NumPyro.
This can be launched as follows:

```
make eval
```

:warning: The complete execution takes hours.

The results of the evaluation are store in the `./logs` directory.
Each log file corresponds to the executions of all the models with a given configuration: backend (Stan/NumPyro) and guide.
Each line of the log files contains the name of the test (model-data), the success status according to the [Stan criteria](https://github.com/stan-dev/performance-tests-cmdstan), the relative error compared to the reference provided by PosteriorDB, the effective sample size, and the raised execution in case of error.
The notebook `analysis.ipynb` is designed to analyze the results.


To select only a subset of the experiments to execute, the Python script `eval.py` can be executed directly:
```
$ python eval.py --help
usage: eval.py [-h] --backend BACKEND --mode MODE [--test]
               [--posteriors POSTERIORS [POSTERIORS ...]] [--guide GUIDE]
               [--steps STEPS] [--samples SAMPLES]

Run autoguide accuracy experiment on PosteriorDB models.

optional arguments:
  -h, --help            show this help message and exit
  --backend BACKEND     inference backend (numpyro, or stan)
  --mode MODE           compilation mode for NumPyro (generative, comprehensive, mixed), algo for Stan (fullrank or meanfield)
  --test                run test experiment (steps = 100, samples = 100)
  --posteriors POSTERIORS [POSTERIORS ...]
                        select the examples to execute
  --guide GUIDE         autoguide (http://num.pyro.ai/en/latest/autoguide.html)
  --steps STEPS         number of svi steps
  --samples SAMPLES     number of samples
```
