import time

import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge, ESN
from reservoirpy.observables import mse, rsquare

import numpy as np
from . import dataset

rpy.verbosity(0)


def model(N, sr, lr, input_scaling, seed, ridge, degree=3, fb=None):
    reservoir = Reservoir(
        N,
        rc_connectivity=degree / N,
        input_connectivity=degree / N,
        fb_connectivity=degree / N,
        sr=sr,
        lr=lr,
        input_scaling=input_scaling,
        fb_scaling=fb if fb is not None else 1.0,
        seed=seed,
    )
    readout = Ridge(ridge=ridge)

    esn = ESN(
        reservoir=reservoir, readout=readout, feedback=(fb is not None), workers=-1
    )

    return esn


def train_model_fold(model, train_test_dataset):
    X_train, Y_train, X_test, Y_test = train_test_dataset

    model.fit(X_train, Y_train, warmup=100)
    return model


def run_model_fold(model, train_test_dataset):
    X_train, Y_train, X_test, Y_test = train_test_dataset

    Y_pred = model.run(X_test)

    return Y_pred


def test_parameters_fold(
    train_test_dataset, N, sr, lr, input_scaling, seed, ridge, degree=3, fb=None
):
    X_train, Y_train, X_test, Y_test = train_test_dataset

    start = time.time()

    esn = model(
        N=N,
        sr=sr,
        lr=lr,
        input_scaling=input_scaling,
        seed=seed,
        ridge=ridge,
        degree=degree,
        fb=fb,
    )
    esn = train_model_fold(model=esn, train_test_dataset=train_test_dataset)
    Y_pred = run_model_fold(model=esn, train_test_dataset=train_test_dataset)

    stop = time.time()

    mses = [mse(y_pred=y_pred, y_true=y_test) for y_pred, y_test in zip(Y_pred, Y_test)]
    r2s = [
        rsquare(y_pred=y_pred, y_true=y_test) for y_pred, y_test in zip(Y_pred, Y_test)
    ]

    return {
        "mses": mses,
        "r2s": r2s,
        "time": stop - start,
    }


def test_parameters_kfold(
    N, sr, lr, input_scaling, seed, ridge, degree=3, fb=None, only_first=False
):
    results = {"mses": [], "r2s": [], "time": []}

    for X_train, Y_train, X_test, Y_test in dataset.kfold():
        fold_result = test_parameters_fold(
            train_test_dataset=(X_train, Y_train, X_test, Y_test),
            N=N,
            sr=sr,
            lr=lr,
            input_scaling=input_scaling,
            seed=seed,
            ridge=ridge,
            degree=degree,
            fb=fb,
        )
        results = {
            "mses": results["mses"] + fold_result["mses"],
            "r2s": results["r2s"] + fold_result["r2s"],
            "time": results["time"] + [fold_result["time"]],
        }
        if only_first:
            break

    return results


def objective(
    dataset,
    config,
    *,
    N,
    sr,
    lr,
    input_scaling,
    ridge,
    seed,
    degree=3,
    fb=None,
    only_first=False,
):
    N = int(N)

    instances = config["instances_per_trial"]
    variable_seed = seed
    losses = []
    r2s = []
    times = []

    for n in range(instances):
        instance_score = test_parameters_kfold(
            N=N,
            sr=sr,
            lr=lr,
            input_scaling=input_scaling,
            seed=seed + n,
            ridge=ridge,
            degree=degree,
            fb=fb,
            only_first=only_first,
        )
        losses += instance_score["mses"]
        r2s += instance_score["r2s"]
        times.append(instance_score["time"])

    return {"loss": np.mean(losses), "r2": np.mean(r2s), "time": np.mean(times)}
