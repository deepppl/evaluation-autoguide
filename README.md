# evaluation-autoguide

Launch the evaluation with:
```
$ python eval.py --help
usage: eval.py [-h] --backend BACKEND --mode MODE [--test] [--posteriors POSTERIORS [POSTERIORS ...]] [--guide GUIDE] [--steps STEPS] [--samples SAMPLES]

Run autoguide accuracy experiment on PosteriorDB models.

optional arguments:
  -h, --help            show this help message and exit
  --backend BACKEND     inference backend (pyro, numpyro, or stan)
  --mode MODE           compilation mode for NumPyro (generative, comprehensive, mixed), algo for Stan (fullrank or meanfield)
  --test                run test experiment (steps = 100, samples = 100)
  --posteriors POSTERIORS [POSTERIORS ...]
                        select the examples to execute
  --guide GUIDE         autoguide (http://num.pyro.ai/en/latest/autoguide.html)
  --steps STEPS         number of svi steps
  --samples SAMPLES     number of samples
```
