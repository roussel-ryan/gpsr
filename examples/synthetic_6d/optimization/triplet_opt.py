# optimizes triplet for minimal beam size when scan elements are off

from typing import Callable, Dict

import numpy as np

import pandas as pd
import torch
from xopt import Evaluator, VOCS, Xopt
from xopt.generators.bayesian import ExpectedImprovementGenerator


def change_triplet_params(lat, k1, k2, k3):
    lat.elements[0].K1.data = k1
    lat.elements[2].K1.data = k2
    lat.elements[4].K1.data = k3


def output(input, beam, lattice):
    """
    calculate final beam sizes as a function of quadrupole strengths [k1,k2,k3]

    Parameters
    ----------
    input: dict with quad strengths
    beam: upstream beam
    lattice: lattice

    Return
    ------
    dict with beamsizes
    """

    # triplet strengths
    K = torch.tensor(np.array([input[f"k{i}"] for i in range(1, 4)]))

    # update lattice
    change_triplet_params(lattice, *K)

    # output beam
    final_beam = lattice(beam)

    # outputs
    std_x = final_beam.x.std()
    std_y = final_beam.y.std()
    # total_size = torch.sqrt((std_x-0.001)**2 + (std_y-0.001)**2)
    total_size = torch.sqrt(std_x**2 + std_y**2) + 1.0 * torch.abs(std_x - std_y)

    return {
        "std_x": float(std_x),
        "std_y": float(std_y),
        "total_size": float(total_size),
    }


def optimize_function(
    vocs: VOCS,
    evaluator_function: Callable,
    n_iterations: int = 5,
    n_initial: int = 5,
    function_kwargs=None,
    generator_kwargs: Dict = None,
) -> Xopt:
    """
    Function to minimize a given function using Xopt's ExpectedImprovementGenerator.

    Details:
    - Initializes BO with a set number of random evaluations given by `n_initial`
    - Raises errors if they occur during calls of `evaluator_function`
    - Runs the generator for `n_iteration` steps
    - Identifies and re-evaluates the best observed point

    Parameters
    ----------
    vocs: VOCS
        Xopt style VOCS object to describe the optimization problem

    evaluator_function : Callable
        Xopt style callable function that is evaluated during the optimization run.

    n_iterations : int, optional
        Number of optimization steps to run. Default: 5

    n_initial : int, optional
        Number of initial random samples to take before performing optimization
        steps. Default: 5

    generator_kwargs : dict, optional
        Dictionary passed to generator to customize Expected Improvement BO.

    Returns
    -------
    X : Xopt
        Xopt object containing the evaluator, generator, vocs and data objects.

    """

    # set up Xopt object
    generator_kwargs = generator_kwargs or {}
    beamsize_evaluator = Evaluator(
        function=evaluator_function, function_kwargs=function_kwargs
    )
    generator = ExpectedImprovementGenerator(vocs=vocs, **generator_kwargs)

    X = Xopt(vocs=vocs, generator=generator, evaluator=beamsize_evaluator)

    # evaluate random intial points
    X.random_evaluate(n_initial)

    # run optimization
    for i in range(n_iterations):
        if i % 5 == 0:
            print(i)
        X.step()

    # get best config and re-evaluate it
    best_config = X.data[X.vocs.variable_names + X.vocs.constant_names].iloc[
        np.argmin(X.data[X.vocs.objective_names].to_numpy())
    ]
    X.evaluate_data(pd.DataFrame(best_config.to_dict(), index=[1]))

    return X
