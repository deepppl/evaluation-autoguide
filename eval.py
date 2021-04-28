import logging, datetime, os, sys, traceback, re, argparse
import numpyro
import jax
from stannumpyro.dppl import NumPyroModel
from numpyro.infer import Trace_ELBO
from numpyro.optim import Adam
import numpyro.infer.autoguide as autoguide
from utils import (
    compile_model,
    get_posterior,
    summary,
    golds,
)
from cmdstanpy import CmdStanModel
import pandas as pd
import numpy


logger = logging.getLogger(__name__)


def run_advi(*, posterior, mode, num_steps, num_samples):
    model = posterior.model
    data = posterior.data.values()
    stanfile = model.code_file_path("stan")
    sm = CmdStanModel(stan_file=stanfile)
    fit = sm.variational(
        iter=num_steps, algorithm=mode, output_samples=num_samples, data=data
    )
    return fit


def run_svi(*, posterior, backend, mode, Autoguide, num_steps, num_samples):
    """
    Compile and run the model.
    Returns the summary Dataframe
    """
    model = posterior.model
    data = posterior.data.values()
    stanfile = model.code_file_path("stan")
    build_dir = f"_build_{backend}_{mode}"
    numpyro_model = NumPyroModel(stanfile, recompile=False, build_dir=build_dir)
    optim = Adam(step_size=0.0005)
    loss = Trace_ELBO()
    guide = Autoguide(numpyro_model.module.model)
    svi = numpyro_model.svi(optim, loss, guide)
    svi.run(jax.random.PRNGKey(0), num_steps, num_samples, data)
    return svi


def compare(*, posterior, backend, mode, Autoguide, num_steps, num_samples, logfile):
    """
    Compare gold standard with model.
    """
    logger.info(f"Processing {posterior.name}")
    sg = summary(posterior.reference_draws())
    if backend == "stan":
        fit = run_advi(
            posterior=posterior, mode=mode, num_steps=num_steps, num_samples=num_samples
        )
        samples = {
            k: numpy.array(fit.variational_sample[i])
            for i, k in enumerate(fit.column_names)
        }
        sm = summary(samples)
        sm = sm[~sm.index.str.endswith("__")]
        sm = sm.rename(columns={"Mean": "mean", "StdDev": "std", "N_Eff": "n_eff"})
    else:
        svi = run_svi(
            posterior=posterior,
            backend=backend,
            mode=mode,
            Autoguide=Autoguide,
            num_steps=num_steps,
            num_samples=num_samples,
        )
        sm = svi.summary()
    sm = sm[["mean", "std", "n_eff"]]
    sm["err"] = abs(sm["mean"] - sg["mean"])
    sm["rel_err"] = sm["err"] / sg["std"]
    if len(sm.dropna()) != len(sg):
        raise RuntimeError("Missing parameter")
    # perf_cmdstan condition: err > 0.0001 and (err / stdev) > 0.3
    comp = sm[(sm["err"] > 0.0001) & (sm["rel_err"] > 0.3)].dropna()
    if not comp.empty:
        logger.error(f"Failed {posterior.name}")
        print(f"{name},mismatch,{sm['n_eff'].mean()}", file=logfile, flush=True)
    else:
        logger.info(f"Success {posterior.name}")
        print(f"{name},success,{sm['n_eff'].mean()}", file=logfile, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run autoguide accuracy experiment on PosteriorDB models."
    )
    parser.add_argument(
        "--backend",
        help="inference backend (pyro, numpyro, or stan)",
        required=True,
    )
    parser.add_argument(
        "--mode",
        help="compilation mode for NumPyro (generative, comprehensive, mixed), algo for Stan (fullrank or meanfield)",
        required=True,
    )

    parser.add_argument(
        "--test",
        help="run test experiment (steps = 100, samples = 100)",
        action="store_true",
    )

    parser.add_argument(
        "--posteriors", nargs="+", help="select the examples to execute"
    )

    parser.add_argument(
        "--guide",
        help="autoguide (http://num.pyro.ai/en/latest/autoguide.html)",
        default="AutoNormal",
    )

    # Override posteriorDB configs
    parser.add_argument("--steps", type=int, help="number of svi steps")
    parser.add_argument("--samples", type=int, help="number of samples")

    args = parser.parse_args()

    if args.posteriors:
        assert all(p in golds for p in args.posteriors), "Bad posterior name"
        golds = args.posteriors

    logging.basicConfig(level=logging.INFO)

    numpyro.set_host_device_count(20)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    today = datetime.datetime.now()
    logpath = f"logs/status_svi_{args.backend}_{args.mode}"
    if args.backend != "stan":
        logpath += f"_{args.guide}"
    logpath += f"_{today.strftime('%y%m%d_%H%M%S')}.csv"
    with open(logpath, "a") as logfile:
        print(",status,n_eff,exception", file=logfile, flush=True)
        for name in (n for n in golds):
            # Configurations
            posterior = get_posterior(name)
            if args.test:
                args.steps = 100
                args.samples = 100
            if args.steps is None:
                args.steps = 100000
            if args.samples is None:
                args.samples = posterior.reference_draws_info()["diagnostics"]["ndraws"]
            try:
                # Compile
                compile_model(posterior=posterior, backend=args.backend, mode=args.mode)
                # Run and Compare
                compare(
                    posterior=posterior,
                    backend=args.backend,
                    mode=args.mode,
                    Autoguide=getattr(autoguide, args.guide),
                    num_steps=args.steps,
                    num_samples=args.samples,
                    logfile=logfile,
                )
            except:
                exc_type, exc_value, _ = sys.exc_info()
                err = " ".join(traceback.format_exception_only(exc_type, exc_value))
                err = re.sub(r"[\n\r\",]", " ", err)[:150] + "..."
                logger.error(f"Failed {name} with {err}")
                print(f'{name},error,,"{err}"', file=logfile, flush=True)
