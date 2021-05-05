help:
	@echo "Help:"
	@echo "  make init         # install the dependencies"
	@echo "  make eval         # run the full evaluation"
	@echo "  make scaled       # run a lighter version of the evaluation"
	@echo "  make clean        # remove the generated files excpet the logs"
	@echo "  make cleanall     # remove all the generated files"

.PHONY: help scaled eval clean cleanall


init:
	git submodule init && git submodule update
	pip install ./posteriordb/python
	opam pin -y -k git git+https://github.com/deepppl/stanc3.git
	pip install -r requirements.txt

eval:
	python eval.py --backend numpyro --mode comprehensive --guide AutoBNAFNormal
	python eval.py --backend numpyro --mode comprehensive --guide AutoDiagonalNormal
	python eval.py --backend numpyro --mode comprehensive --guide AutoMultivariateNormal
	python eval.py --backend numpyro --mode comprehensive --guide AutoIAFNormal
	python eval.py --backend numpyro --mode comprehensive --guide AutoLaplaceApproximation
	python eval.py --backend numpyro --mode comprehensive --guide AutoLowRankMultivariateNormal
	python eval.py --backend numpyro --mode comprehensive --guide AutoNormal
	python eval.py --backend numpyro --mode comprehensive --guide AutoDelta
	python eval.py --backend stan --mode meanfield
	python eval.py --backend stan --mode fullrank

test:
	python eval.py --backend numpyro --mode comprehensive --test

clean:
	rm -rf _tmp __pycache__ _build*

cleanall: clean
	rm -rf logs *~
