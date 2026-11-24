"""Models for optimizing detector geometry using PyFAI and Bayesian optimization.

Classes:
    - BayFAIParameters:
        Parameters for running BayFAI
"""

__all__ = ["BayFAIParameters"]
__author__ = "Louis Conreux"

import os
import warnings
from typing import Any, Dict, List, Literal, Tuple, Optional

import pydantic

from lute.io.models.base import TaskParameters
from lute.io.models.validators import (
    validate_smd_path,
)

pydantic_ver: int = int(pydantic.version.VERSION[0])
if pydantic_ver > 1:
    from pydantic.v1 import BaseModel, Field, validator  # type: ignore
else:
    from pydantic import BaseModel, Field, validator


def validate_geometry_path(output_path_name: str):
    """Dynamically generates the output geometry path for the optimization results."""

    def _validate_geometry_path(cls, output_path: str, values: Dict[str, Any]) -> str:
        if output_path == "":
            work_dir = values["lute_config"].work_dir
            run = int(values["lute_config"].run)
            detname = values["detname"]
            geom_dir = os.path.join(work_dir, "geom")
            os.makedirs(geom_dir, exist_ok=True)
            output_run_path = os.path.join(geom_dir, f"{run}-end_{detname}.data")
            return output_run_path
        return output_path

    return validator(output_path_name, always=True)(_validate_geometry_path)


def validate_bounds(bounds_name: str):
    """Rejects bounds that are not absolute [low, high] limits."""

    def _validate_bounds(
        cls, bounds: Dict[str, Tuple[float, float]], values: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        errors: List[str] = []
        for p, (low, high) in bounds.items():
            if low >= high:
                errors.append(f"{p}: low {low} >= high {high}")
            elif p == "dist" and low <= 0:
                errors.append(f"dist: low {low} <= 0")
        if errors:
            raise ValueError(
                "; ".join(errors) + ". BayFAI bounds are absolute [low, high] limits."
            )
        return bounds

    return validator(bounds_name, always=True, allow_reuse=True)(_validate_bounds)


def validate_center(center_name: str):
    """Defaults a missing or out-of-bounds free center to its bounds midpoint."""

    def _validate_center(
        cls, center: Dict[str, float], values: Dict[str, Any]
    ) -> Dict[str, float]:
        bounds: Dict[str, Tuple[float, float]] = values.get("bounds", {})
        fixed: List[str] = values.get("fixed", [])
        center = dict(center)
        for p, (low, high) in bounds.items():
            if p not in center or (p not in fixed and not low <= center[p] <= high):
                if p in center:
                    warnings.warn(
                        f"BayFAI {p}: center {center[p]} outside [{low}, {high}], "
                        "using midpoint"
                    )
                center[p] = (low + high) / 2
        return center

    return validator(center_name, always=True, allow_reuse=True)(_validate_center)


class BayFAIParameters(TaskParameters):
    """Parameters for optimizing detector geometry using PyFAI and Bayesian optimization.

    BayFAI has default hyperparameters that can be overriden by the user.
    """

    class Config(TaskParameters.Config):
        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

    class BayFAIHyperparameters(BaseModel):
        """BayFAI hyperparameters."""

        n_samples: int = Field(
            default=20,
            description="Number of samples to initialize the Gaussian Process.",
        )

        n_iterations: int = Field(
            default=40,
            description="Number of iterations of Bayesian Optimization",
        )

        max_rings: int = Field(
            default=8,
            description="Maximum number of rings to search for Bragg peaks.",
        )

        pts_per_deg: float = Field(
            default=2.0,
            description="Number of Bragg peaks to extract per azimuthal degree.",
        )

        Imin: float = Field(
            default=95,
            description="Minimum intensity percentile threshold for Bragg peak detection.",
        )

        prior: bool = Field(
            default=True,
            description="Whether to sample initial points around the center of search space or randomly.",
        )

        beta: float = Field(
            default=1.96,
            description="Exploration-exploitation trade-off hyperparameter for Upper Confidence Bound acquisition function.",
        )

        step: int = Field(
            default=5,
            description="Size of the refinement space around best parameters.",
        )

        lbda: float = Field(
            default=0.1,
            description="Penalty weight in final scoring to select winner geometry.",
        )

        seed: Optional[int] = Field(
            default=None,
            description="Random seed for reproducibility.",
        )

        n_cand: Optional[int] = Field(
            default=8192,
            description="Number of random grid points scored by UCB per BO step (0 or None = whole grid).",
        )

        gp_restarts: int = Field(
            default=10,
            description="n_restarts_optimizer of the GP kernel-hyperparameter fit (sklearn GaussianProcessRegressor).",
        )

        min_points: Optional[int] = Field(
            default=None,
            description="Geometries with fewer extracted control points score 0 (0 = off; "
            "None = 100 when parallelized is 'dist', else 0).",
        )

    parallelized: Optional[str] = Field(
        None,
        description="Name of the geometry parameter to distribute across MPI ranks as a "
        "sliding window (e.g. 'dist'). If None, all ranks instead run BO over the same space.",
    )

    dist_spacing: Literal["geometric", "linear"] = Field(
        "geometric",
        description="With parallelized: dist, how bounds.dist is split across ranks: "
        "'geometric' (window widths grow with distance, b = low*(high/low)**(r/size)) or "
        "'linear' (equal windows). Each rank's grid step is its window width.",
    )

    fixed: List[str] = Field(
        ["rot3"],
        description="List of fixed parameters for the optimization (rot3 is fixed by default) "
        "because diffraction rings are invariance by rotation around the beam axis.",
    )

    bounds: Dict[str, Tuple[float, float]] = Field(
        {
            "dist": (0.035, 0.4),
            "poni1": (-0.0025, 0.0025),
            "poni2": (-0.0025, 0.0025),
            "rot1": (-0.1, 0.1),
            "rot2": (-0.1, 0.1),
            "rot3": (-0.1, 0.1),
        },
        description="Absolute [low, high] limits of the search for each geometry parameter "
        "(not offsets around center). With parallelized: dist, bounds.dist is the whole "
        "distance range split across ranks.",
    )

    center: Dict[str, float] = Field(
        {},
        description="Prior mean and starting point for each geometry parameter, and the "
        "value of fixed parameters. Missing parameters, or free parameters outside bounds, "
        "default to the bounds midpoint.",
    )

    resolutions: Dict[str, float] = Field(
        {
            "dist": 0.001,
            "poni1": 0.0001,
            "poni2": 0.0001,
            "rot1": 0.02,
            "rot2": 0.02,
            "rot3": 0.02,
        },
        description="Resolution of the search space for the detector geometry parameters.",
    )

    detname: str = Field(
        "",
        description="Detector name",
    )

    calibrant: str = Field(
        "",
        description="Calibrant used for the calibration supported by pyFAI: https://github.com/silx-kit/pyFAI/tree/main/src/pyFAI/resources/calibration, \n e.g. Silver Behenate 'AgBh', LaB6 'CeO2', etc.",
    )

    wavelength: float = Field(
        1e-10,
        description=(
            "Wavelength in meters. If provided (non-default), it takes precedence "
            "over the mean photon energy read from the h5 file."
        ),
    )

    h5: str = Field(
        "",
        description="Smalldata hdf5 file path to be used for the calibration.",
    )

    out_file: str = Field(
        "",
        description="Path to the output .data file that will contain the optimized detector geometry.",
        is_result=True,
    )

    bayfai_params: BayFAIHyperparameters = Field(
        BayFAIHyperparameters(),
        description="BayFAI hyperparameters.",
    )

    _find_smd_path = validate_smd_path("h5")

    _check_bounds = validate_bounds("bounds")
    _fill_center = validate_center("center")

    _find_out_file_path = validate_geometry_path("out_file")
