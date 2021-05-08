help:
	@echo "Help:"
	@echo "  make init         # install the dependencies"
	@echo "  make eval         # run the full evaluation"
	@echo "  make test         # test the evaluation scripts"
	@echo "  make clean        # remove the generated files excpet the logs"
	@echo "  make cleanall     # remove all the generated files"

.PHONY: help scaled eval clean cleanall

init:
	git submodule init && git submodule update
	pip install ./posteriordb/python
	opam pin -y -k git git+https://github.com/deepppl/stanc3.git
	pip install -r requirements.txt

eval: 	eval-AutoBNAFNormal eval-AutoDiagonalNormal \
	eval-AutoMultivariateNormal eval-AutoIAFNormal \
	eval-AutoLaplaceApproximation eval-AutoLowRankMultivariateNormal \
	eval-AutoNormal eval-AutoDelta \
	eval-stan

eval-stan:
	python eval.py --backend stan --mode meanfield
	python eval.py --backend stan --mode fullrank

eval-%:
	python eval.py --backend numpyro --mode comprehensive --guide $(@:eval-%=%)

test:
	python eval.py --backend numpyro --mode comprehensive --test

clean:
	rm -rf _tmp __pycache__ _build*

cleanall: clean
	rm -rf logs *~
