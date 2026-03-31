"""
Classes for geometry optimization tasks.

Classes:
    BayFAIOpt: optimize LCLS detector geometry using PyFAI coupled with Bayesian Optimization.

Functions:
    Miscellaneous functions for geometry-related calculations:
    - rotation_matrix: Compute and return the detector tilts as a single rotation matrix
    - correct_geom: Correct the geometry given a set of geometry parameters.
    - calculate_radius: Calculate the radius for each pixel based on the geometry parameters.
    - calculate_2theta: Calculate the 2θ angles for each pixel based on the geometry parameters.
    - calculate_q: Calculate the q-vector magnitude for each pixel based on the geometry parameters.
    - azimuthal_integration: Compute the radial intensity profile of an image.
    - theta2q: Convert pixel diffraction angles to scattering vector magnitude q.
    - r2q: Convert pixel radii to scattering vector magnitude q.
"""

__all__ = [
    "BayFAIOpt",
    "rotation_matrix",
    "correct_geom",
    "calculate_radius",
    "calculate_2theta",
    "calculate_q",
    "azimuthal_integration",
    "theta2q",
    "r2q",
]

__author__ = "Louis Conreux"

from lute.execution.logging import get_logger

import os
import numpy as np
import numpy.typing as npt
from typing import Optional
import logging
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as patches  # type: ignore
from matplotlib import lines  # type: ignore
from bokeh.plotting import figure  # type: ignore
from bokeh.models import LinearColorMapper, HoverTool  # type: ignore
from bokeh.palettes import Viridis256  # type: ignore
from bokeh.models.annotations import Label  # type: ignore
import h5py  # type: ignore
import pyFAI  # type: ignore
from pyFAI.geometry import Geometry  # type: ignore
from pyFAI.goniometer import SingleGeometry  # type: ignore
from pyFAI.geometryRefinement import GeometryRefinement  # type: ignore
from pyFAI.calibrant import CALIBRANT_FACTORY  # type: ignore
from pyFAI.units import RADIAL_UNITS  # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # type: ignore
from sklearn.utils._testing import ignore_warnings  # type: ignore
from sklearn.exceptions import ConvergenceWarning  # type: ignore
from mpi4py import MPI

from LCLSGeom.manager import get_geometry  # type: ignore
from LCLSGeom.converter import PsanaToPyFAI, PyFAIToPsana, PyFAIToCrystFEL  # type: ignore

pyFAI.use_opencl = False

logger: logging.Logger = get_logger(__name__)


def rotation_matrix(params: list) -> np.ndarray:
    """
    Compute and return the detector tilts as a single rotation matrix

    Parameters
    ----------
    params : list
        Detector parameters found by PyFAI calibration
    """
    cos_rot1 = np.cos(params[3])
    cos_rot2 = np.cos(params[4])
    cos_rot3 = np.cos(params[5])
    sin_rot1 = np.sin(params[3])
    sin_rot2 = np.sin(params[4])
    sin_rot3 = np.sin(params[5])
    # Rotation about vertical axis: Note this rotation is left-handed
    rot1 = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_rot1, sin_rot1], [0.0, -sin_rot1, cos_rot1]]
    )
    # Rotation about horizontal axis: Note this rotation is left-handed
    rot2 = np.array(
        [[cos_rot2, 0.0, -sin_rot2], [0.0, 1.0, 0.0], [sin_rot2, 0.0, cos_rot2]]
    )
    # Rotation about z-axis: Note this rotation is right-handed
    rot3 = np.array(
        [[cos_rot3, -sin_rot3, 0.0], [sin_rot3, cos_rot3, 0.0], [0.0, 0.0, 1.0]]
    )
    rotation_matrix = np.dot(np.dot(rot3, rot2), rot1)
    return rotation_matrix


def correct_geom(detector: pyFAI.detectors.Detector, params: Optional[list] = None):
    """
    Correct the geometry given a set of geometry parameters.

    Parameters
    ----------
    detector : pyFAI.detectors.Detector
        PyFAI detector object containing pixel coordinates.
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
    """
    x, y, z = detector.calc_cartesian_positions()
    if params is not None:
        dist = params[0]
        poni1 = params[1]
        poni2 = params[2]
        x = (x - poni1).ravel()
        y = (y - poni2).ravel()
        if z is None:
            z = np.zeros_like(x) + dist
        else:
            z = (z + dist).ravel()
        coord_det = np.vstack((x, y, z))
        x, y, z = np.dot(rotation_matrix(params), coord_det)
    else:
        if z is None:
            z = np.zeros_like(x)
    x = np.reshape(x, detector.raw_shape)
    y = np.reshape(y, detector.raw_shape)
    z = np.reshape(z, detector.raw_shape)
    return x, y, z


def calculate_radius(
    detector: pyFAI.detectors.Detector, params: Optional[list] = None
) -> np.ndarray:
    """
    Calculate the radius for each pixel based on the geometry parameters.

    Parameters
    ----------
    detector  : pyFAI.Detector
        pyFAI detector object
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz

    Returns
    -------
    r : numpy.ndarray, with input shape
        map of pixels' radii
    """
    x, y, _ = correct_geom(detector, params)
    r = np.zeros(detector.raw_shape)
    for p in range(detector.n_modules):
        r[p] = np.sqrt(x[p] ** 2 + y[p] ** 2)
    return r


def calculate_2theta(
    detector: pyFAI.detectors.Detector, params: Optional[list] = None
) -> np.ndarray:
    """
    Calculate the 2θ angles for the detector based on the geometry parameters.

    Parameters
    ----------
    detector : pyFAI.detectors.Detector
        PyFAI detector object containing pixel coordinates to be corrected.
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
    """
    x, y, z = correct_geom(detector, params)
    tth = np.zeros(detector.raw_shape)
    for p in range(detector.n_modules):
        tth[p] = np.arctan2(np.sqrt(x[p] * x[p] + y[p] * y[p]), z[p])
    return tth


def calculate_q(
    detector: pyFAI.detectors.Detector, params: Optional[list] = None
) -> np.ndarray:
    """
    Calculate the q-vectors for each pixel based on the geometry parameters.

    Parameters
    ----------
    detector : pyFAI.detectors.Detector
        PyFAI detector object containing pixel coordinates to be corrected.
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
    """
    tth = calculate_2theta(detector, params)
    wavelength = detector.wavelength
    q = 4.0 * np.pi * np.sin(tth / 2.0) / (wavelength * 1e10)
    return q


def azimuthal_integration(
    powder: npt.NDArray[np.float64],
    detector: pyFAI.detectors.Detector,
    params: Optional[list] = None,
) -> tuple:
    """
    Compute the radial intensity profile of an image.

    Parameters
    ----------
    powder : numpy.ndarray, shape (n,m)
        detector image
    detector : pyFAI.Detector
        PyFAI detector object
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
    """
    tth = calculate_2theta(detector, params)
    nbins = 1024
    intensity, bin_edges = np.histogram(
        tth.ravel(), bins=nbins, range=(tth.min(), tth.max()), weights=powder.ravel()
    )
    count, _ = np.histogram(tth.ravel(), bins=bin_edges)
    radialprofile = np.divide(
        intensity, count, out=np.zeros_like(intensity), where=count != 0
    )
    tth_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return radialprofile, tth_centers


def theta2q(theta: np.ndarray, wavelength: float) -> np.ndarray:
    """
    Convert pixel 2θ angles to scattering vector magnitude q.

    Parameters
    ----------
    theta: : numpy.ndarray, 1d
        diffraction angles in radians
    wavelength : float
        X-ray wavelength in Angstrom

    Returns
    -------
    qs: numpy.ndarray, 1d
        magnitude of q-vector in per Angstrom
    """
    qs = 4.0 * np.pi * np.sin(theta / 2.0) / (wavelength * 1e10)
    return qs


def r2q(radii: np.ndarray, distance: float, wavelength: float) -> np.ndarray:
    """
    Convert pixel radii to scattering vector magnitude q.

    Parameters
    ----------
    radii : numpy.ndarray, 1d
        radius in meter from beam center
    distance : float
        detector distance in meter
    wavelength : float
        X-ray wavelength in Angstrom

    Returns
    -------
    qs: numpy.ndarray, 1d
        magnitude of q-vector in per Angstrom
    """
    theta = np.arctan2(radii, distance)
    qs = theta2q(theta, wavelength)
    return qs


class BayFAIOpt:
    """
    Class to run BayFAI optimization on a powder image.

    Parameters
    ----------
    exp : str
        Experiment name
    run : int
        Run number
    """

    def __init__(
        self,
        exp,
        run,
    ):
        self.exp = exp
        self.run = run
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        if self.rank == 0:
            logger.info(f"Getting {self.size} processes for BayFAIOpt task")

    @staticmethod
    def UCB(X, gp_model, visited_idx, beta=1.96):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        ucb = y_pred + beta * y_std
        ucb[visited_idx] = -np.inf
        next = np.argmax(ucb)
        return next

    @staticmethod
    def q_UCB(X, gp_model, q, visited_idx, beta=1.96):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        ucb = y_pred + beta * y_std
        ucb[visited_idx] = -np.inf
        top_next = np.argsort(ucb)[-q:]
        return top_next

    def setup(
        self,
        fixed: list,
        detname: str,
        h5: str,
        smooth: bool,
        calibrant: str,
        wavelength: float,
    ):
        """
        Setup the BayFAI optimization.

        Parameters
        ----------
        fixed : list
            List of parameters to keep fixed during optimization
        detname : str
            Name of the detector
        h5 : str
            Path to the h5 file to use for calibration
        smooth : bool
            If True, apply smoothing to the powder image
        calibrant : PyFAI.Calibrant
            PyFAI calibrant object
        wavelength : float
            X-ray wavelength in meters

        Returns
        -------
        Imin : float
            Minimum intensity value for identifying Bragg peaks
        """
        self.detector = self.build_detector(detname)
        self.powder = self.generate_powder(h5, detname, smooth)
        self.stacked_powder = np.reshape(self.powder, self.detector.shape)
        pos_pix = self.powder[self.powder > 0]
        self.Imin = np.percentile(pos_pix, 98)
        self.calibrant = self.define_calibrant(calibrant, h5, wavelength)
        self.set_search_space(fixed)

    def extract_powder(self, powder_path: str, detname: str) -> npt.NDArray[np.float64]:
        """
        Extract a powder image from smalldata analysis.

        Parameters
        ----------
        powder_path : str
            Path to the h5 or npy file containing the powder data.

        Returns
        -------
        powder : npt.NDArray[np.float64]
            The extracted powder image.
        """
        if powder_path.endswith(".npy"):
            powder = np.load(powder_path)
            return powder
        else:
            with h5py.File(powder_path) as h5:
                try:
                    powder = h5[f"Sums/{detname}_calib_max"][()]
                except KeyError:
                    logger.warning(
                        f"Cannot find {detname} Max powder in {powder_path}, defaulting to {detname} Sum instead."
                    )
                    try:
                        powder = h5[f"Sums/{detname}_calib"][()]
                    except KeyError:
                        logger.error(
                            f"Cannot find {detname} Sum powder in {powder_path}. Exiting..."
                        )
                        raise
            return powder

    def preprocess_powder(
        self,
        powder: npt.NDArray[np.float64],
        mask: npt.NDArray[np.integer],
        smooth: bool = False,
    ) -> npt.NDArray[np.float64]:
        """
        Preprocess extracted powder for enhancing optimization

        Parameters
        ----------
        powder : npt.NDArray[np.float64]
            Powder image to use for calibration
        mask : npt.NDArray[np.integer]
            Pixel mask to apply to the powder image
        smooth : bool, optional
            If True, apply smoothing to the powder image.
        """
        powder[powder < 0] = 0
        if smooth:
            for p in range(powder.shape[0]):
                gradx = np.gradient(powder[p], axis=0)
                grady = np.gradient(powder[p], axis=1)
                powder[p] = np.sqrt(gradx**2 + grady**2)
        powder[mask == 0] = 0
        return powder

    def assemble_image(
        self, powder: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Assemble the powder image from modules to full detector shape.

        Parameters
        ----------
        powder : npt.NDArray[np.float64]
            Powder image to use for calibration
        """
        pixel_index_map = self.detector.pixel_index_map
        max_rows = np.max(pixel_index_map[..., 0]) + 1
        max_cols = np.max(pixel_index_map[..., 1]) + 1
        assembled_powder = np.zeros((max_rows, max_cols))
        for p in range(pixel_index_map.shape[0]):
            i = pixel_index_map[p, ..., 0]
            j = pixel_index_map[p, ..., 1]
            assembled_powder[i, j] = powder[p]
        return assembled_powder

    def generate_powder(
        self, powder_path: str, detname: str, smooth: bool = False
    ) -> npt.NDArray[np.float64]:
        """
        Generate a preprocessed powder image from smalldata reduction.

        Parameters
        ----------
        powder_path : str
            Path to the h5 or npy file containing the powder data.
        detname : str
            Name of the detector
        smooth : bool, optional
            If True, apply smoothing to the powder image.
        """
        mask = self.detector.geo.get_pixel_mask(mbits=3)
        mask = np.squeeze(mask)
        powder = self.extract_powder(powder_path, detname)
        powder = self.preprocess_powder(powder, mask, smooth)
        self.assembled_powder = self.assemble_image(powder)
        return powder

    def build_detector(
        self, detname: str,
    ) -> pyFAI.detectors.Detector:
        """
        Read the metrology data and build a pyFAI detector object.

        Parameters
        ----------
        detname : str
            Name of the detector

        Returns
        -------
        pyFAI.Detector
            Configured pyFAI detector object
        """
        in_file = get_geometry(detname)
        detector = PsanaToPyFAI.convert(in_file, detname)
        return detector

    def update_geometry(self, out_file: str, detname: str) -> pyFAI.detectors.Detector:
        """
        Update the geometry and write a new .poni, .geom and .data file

        Parameters
        ----------
        out_file : str
            Path to the output file
        detname : str
            Name of the detector
        """
        path = os.path.dirname(out_file)
        poni_file = os.path.join(path, f"r{self.run:0>4}.poni")
        self.gr.save(poni_file)
        PyFAIToPsana.convert(
            in_file=poni_file,
            detector=self.detector,
            out_file=out_file,
        )
        geom_file = os.path.join(path, f"r{self.run:0>4}.geom")
        PyFAIToCrystFEL.convert(
            in_file=poni_file,
            detector=self.detector,
            out_file=geom_file,
        )
        detector = PsanaToPyFAI.convert(
            in_file=out_file,
            detname=detname,
        )
        return detector

    def define_calibrant(
        self,
        h5: str,
        calibrant_name: str,
    ) -> pyFAI.calibrant.Calibrant:
        """
        Define calibrant for optimization with appropriate wavelength

        Parameters
        ----------
        h5 : str
            Path to the h5 file containing the wavelength data
        calibrant_name : str
            Name of the calibrant
        wavelength : float
            X-ray wavelength in meters
        """
        self.calibrant_name = calibrant_name
        calibrant = CALIBRANT_FACTORY(calibrant_name)
        try:
            with h5py.File(h5) as f:
                if "ebeam" in f:
                    ebeam_key = "ebeam"
                elif "ebeamh" in f:
                    ebeam_key = "ebeamh"
                else:
                    raise KeyError("Neither 'ebeam' nor 'ebeamh' found in h5 file")
                photon_energy = np.mean(f[ebeam_key]["photon_energy"][()])
                wavelength = 1.23984193e-6 / photon_energy
        except Exception as e:
            logger.warning(f"Could not read photon energy from {h5} due to {e}, defaulting to provided wavelength {wavelength} m")
        calibrant.wavelength = wavelength
        return calibrant

    def set_search_space(self, fixed: list) -> None:
        """
        Define the search space for the free parameters.

        Parameters
        ----------
        fixed : list
            List of parameters to keep fixed during optimization
        """
        self.fixed = fixed
        self.space = []
        parallelized = ["dist"]
        self.order = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        for p in self.order:
            if p not in fixed and p not in parallelized:
                self.space.append(p)

    def distribute_distances(self, center, res):
        """
        Distribute distances across MPI ranks.

        Parameters
        ----------
        center : dict
            Center values for each parameter
        res : float
            Resolution of the grid used to discretize the parameter search space

        Returns
        -------
        dist : float
            The distance assigned to this MPI rank
        """
        half = self.size // 2
        offsets = (np.arange(self.size) - half) * res["dist"]
        distances = center["dist"] + offsets
        distances = np.round(distances, 6)
        self.distances = distances
        dist = distances[self.rank]
        return dist

    def create_search_space(self, dist, center, bounds, res):
        """
        Discretize the search space for the free parameters.

        Parameters
        ----------
        dist : float
            Distance on this MPI rank
        center : dict
            Center values for each parameter
        bounds : dict
            Bounds for each parameter, format: {param: (lower, upper)}
        res : dict
            Resolution per parameter

        Returns
        -------
        X : np.ndarray
            Full 6D geometry space (cartesian product)
        X_norm : np.ndarray
            Normalized search space (between-1 and 1)
        """
        center["dist"] = dist
        full_params = {}
        search_params = {}
        for p in self.order:
            if p in self.space:
                low = center[p] + bounds[p][0]
                high = center[p] + bounds[p][1]
                if high < low:
                    low, high = high, low
                step = res[p]
                full_params[p] = np.arange(low, high + step, step)
                search_params[p] = full_params[p]
            else:
                full_params[p] = np.array([center[p]])

        X = np.array(np.meshgrid(*[full_params[p] for p in self.order])).T.reshape(
            -1, len(self.order)
        )
        X_search = np.array(
            np.meshgrid(*[search_params[p] for p in self.space])
        ).T.reshape(-1, len(self.space))
        self.mins = np.min(X_search, axis=0)
        self.maxs = np.max(X_search, axis=0)
        X_norm = 2 * (X_search - self.mins) / (self.maxs - self.mins) - 1
        return X, X_norm

    def sample_initial_points(self, X, X_norm, center, bounds, n_samples, prior):
        """
        Sample initial points from the search space.

        Parameters
        ----------
        X : np.ndarray
            Search space
        X_norm : np.ndarray
            Normalized search space
        center : dict
            Center values for each parameter
        bounds : dict
            Bounds for each parameter
        n_samples : int
            Number of samples to draw
        prior : bool
            Use prior information for sampling

        Returns
        -------
        np.ndarray
            Sampled points
        """
        if prior:
            means = [center[p] for p in self.space]
            cov = np.diag(
                [(np.abs((bounds[p][1] - bounds[p][0])) / 5) ** 2 for p in self.space]
            )
            X_free = np.random.multivariate_normal(means, cov, n_samples)
            X_free = np.clip(X_free, self.mins, self.maxs)
            X_norm_samples = 2 * (X_free - self.mins) / (self.maxs - self.mins) - 1
            X_samples = np.tile([center[p] for p in self.order], (n_samples, 1))
            for i, p in enumerate(self.space):
                j = self.order.index(p)
                X_samples[:, j] = X_free[:, i]
            return X_samples, X_norm_samples
        else:
            idx_samples = np.random.choice(X.shape[0], n_samples)
            X_samples = X[idx_samples]
            X_norm_samples = X_norm[idx_samples]
            return X_samples, X_norm_samples

    def score(self, sample, Imin, max_rings):
        """
        Evaluate score at a given sampled geometry based on the residual between predicted and observed Bragg peak positions.

        Parameters
        ----------
        sample : list
            Geometry parameters
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider

        Returns
        -------
        score : float
            Scalar score for Bayesian optimization
        """
        dist, poni1, poni2, rot1, rot2, rot3 = sample
        geom_sample = Geometry(
            dist=dist,
            poni1=poni1,
            poni2=poni2,
            rot1=rot1,
            rot2=rot2,
            rot3=rot3,
            detector=self.detector,
            wavelength=self.calibrant.wavelength,
        )
        sg = SingleGeometry(
            "Score Geometry",
            self.stacked_powder,
            calibrant=self.calibrant,
            detector=self.detector,
            geometry=geom_sample,
        )
        sg.extract_cp(max_rings=max_rings, pts_per_deg=1, Imin=Imin)
        data = sg.geometry_refinement.data

        if data is None or len(data) == 0:
            return 0.0

        ix = data[:, 0]
        iy = data[:, 1]
        ring = data[:, 2].astype(np.int32)
        score = -np.log(
            sg.geometry_refinement.residu2(sample, ix, iy, ring) / len(data)
        )
        return score

    def estimate_uncertainty(self, refinement, rel_eps=1e-3, abs_eps=1e-4):
        """
        Estimate parameter uncertainties from the Hessian matrix.

        Parameters
        ----------
        refinement : GeometryRefinement
            pyFAI refinement object after refine3.
        rel_eps : float
            Relative step for finite differences.
        abs_eps : float
            Absolute step for finite differences.

        Returns
        -------
        sigmas : np.ndarray
            Estimated uncertainties for each parameter
        is_min : bool
            True if a local minimum was found
        """
        param0 = np.array(
            [
                refinement.dist,
                refinement.poni1,
                refinement.poni2,
                refinement.rot1,
                refinement.rot2,
                refinement.rot3,
            ],
            dtype=np.float64,
        )
        param_names = ["dist", "poni1", "poni2", "rot1", "rot2"]
        size = len(param_names)

        d1 = refinement.data[:, 0]
        d2 = refinement.data[:, 1]
        ring = refinement.data[:, 2].astype(np.int32)
        f_min = refinement.residu2(param0, d1, d2, ring)
        hessian = np.zeros((size, size), dtype=np.float64)
        dof = max(len(refinement.data) - size, 1)

        delta = np.maximum(rel_eps * np.abs(param0), abs_eps)
        for i in range(size):
            deltai = delta[i]
            param = param0.copy()
            param[i] += deltai
            f_plus = refinement.residu2(param, d1, d2, ring)
            param = param0.copy()
            param[i] -= deltai
            f_minus = refinement.residu2(param, d1, d2, ring)
            hessian[i, i] = (f_plus + f_minus - 2.0 * f_min) / (deltai**2)

            for j in range(i + 1, size):
                deltaj = delta[j]
                param = param0.copy()
                param[i] += deltai
                param[j] += deltaj
                f_pp = refinement.residu2(param, d1, d2, ring)
                param = param0.copy()
                param[i] -= deltai
                param[j] -= deltaj
                f_mm = refinement.residu2(param, d1, d2, ring)
                param = param0.copy()
                param[i] += deltai
                param[j] -= deltaj
                f_pm = refinement.residu2(param, d1, d2, ring)
                param = param0.copy()
                param[i] -= deltai
                param[j] += deltaj
                f_mp = refinement.residu2(param, d1, d2, ring)
                hessian[j, i] = hessian[i, j] = (f_pp + f_mm - f_pm - f_mp) / (
                    4.0 * deltai * deltaj
                )

        eigs, _ = np.linalg.eigh(hessian)
        if np.any(eigs <= 0):
            sigmas = [np.inf] * size
            is_min = False
            penalty = 0.0
            return sigmas, is_min, penalty

        cov = np.linalg.inv(hessian)
        sigmas = f_min * np.diag(cov) / dof
        sigmas = np.sqrt(sigmas)
        is_min = True
        penalty = -np.log(np.linalg.det(cov)) / 2
        return sigmas, is_min, penalty

    def gradient_descent(self, best_param, resolutions, Imin, max_rings, step=5):
        """
        Evaluate geometry found by BO on pyFAI refinement tool

        Parameters
        ----------
        best_param : list
            Best parameters found by Bayesian optimization
        resolutions : dict
            Resolution per parameter for restricted refinement
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider
        step : int
            Size of the refinement space around best parameters

        Returns
        -------
        score : float
            Negative log of the residual after refinement
        sigma : np.ndarray
            Estimated uncertainties for each parameter
        penalty : float
            Penalty from uncertainty estimation
        size : int
            Number of Bragg peaks used in refinement
        params : dict
            Refined parameters
        is_min : bool
            Flag indicating if a local minimum was found
        """
        dist, poni1, poni2, rot1, rot2, rot3 = best_param
        best_geom = Geometry(
            dist=dist,
            poni1=poni1,
            poni2=poni2,
            rot1=rot1,
            rot2=rot2,
            rot3=rot3,
            detector=self.detector,
            wavelength=self.calibrant.wavelength,
        )
        sg = SingleGeometry(
            "Best Geometry",
            self.stacked_powder,
            calibrant=self.calibrant,
            detector=self.detector,
            geometry=best_geom,
        )
        sg.extract_cp(max_rings=max_rings, pts_per_deg=1, Imin=Imin)
        self.sg = sg

        if sg.geometry_refinement.data is None or len(sg.geometry_refinement.data) == 0:
            score = 0.0
            sigma = [np.inf] * 5
            penalty = 0.0
            size = 0
            is_min = False
            return score, sigma, penalty, size, best_param, is_min

        sg.geometry_refinement.set_dist_min(dist - step * resolutions["dist"])
        sg.geometry_refinement.set_dist_max(dist + step * resolutions["dist"])
        sg.geometry_refinement.set_poni1_min(poni1 - step * resolutions["poni1"])
        sg.geometry_refinement.set_poni1_max(poni1 + step * resolutions["poni1"])
        sg.geometry_refinement.set_poni2_min(poni2 - step * resolutions["poni2"])
        sg.geometry_refinement.set_poni2_max(poni2 + step * resolutions["poni2"])
        sg.geometry_refinement.set_rot1_min(rot1 - step * resolutions["rot1"])
        sg.geometry_refinement.set_rot1_max(rot1 + step * resolutions["rot1"])
        sg.geometry_refinement.set_rot2_min(rot2 - step * resolutions["rot2"])
        sg.geometry_refinement.set_rot2_max(rot2 + step * resolutions["rot2"])
        fix = ["rot3", "wavelength"]
        score = -np.log(sg.geometry_refinement.refine3(fix=fix))
        sigma, is_min, penalty = self.estimate_uncertainty(sg.geometry_refinement)
        params = sg.geometry_refinement.param
        size = len(sg.geometry_refinement.data)
        return score, sigma, penalty, size, params, is_min

    @ignore_warnings(category=ConvergenceWarning)
    def bayes_opt_distance(
        self,
        dist,
        center,
        bounds,
        res,
        n_samples,
        n_iterations,
        Imin,
        max_rings,
        beta=1.96,
        step=5,
        prior=True,
        seed=None,
    ):
        """
        Run Bayesian Optimization on a subspace of fixed distance.

        Parameters
        ----------
        dist : float
            Distance on this MPI rank
        center : dict
            Dictionary of center values for each parameter
        bounds : dict
            Dictionary of bounds for each parameter
        res : dict
            Dictionary of resolution for each parameter
        n_samples : int
            Number of samples to initialize the Gaussian Process
        n_iterations : int
            Number of iterations of Bayesian Optimization
        Imin : float
            Minimum intensity threshold for identifying Bragg peaks
        max_rings : int
            Maximum number of rings to search for Bragg peaks
        beta : float
            Exploration-exploitation trade-off parameter for UCB acquisition function
        step : int
            Size of the refinement space around best parameters
        prior : bool
            Whether to sample initial points around the center or randomly
        seed : optional, int
            Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. Create the search space
        X, X_norm = self.create_search_space(dist, center, bounds, res)

        # 2. Sample initial points
        X_samples, X_norm_samples = self.sample_initial_points(
            X, X_norm, center, bounds, n_samples, prior
        )

        # 3. Evaluate the initial points
        bo_history = {"params": [], "scores": []}
        y = np.zeros((n_samples))
        for i in range(n_samples):
            y[i] = self.score(X_samples[i], Imin, max_rings)
            bo_history["params"].append(X_samples[i])
            bo_history["scores"].append(y[i])

        if np.all(y == 0.0):
            result = {
                "bo_history": bo_history,
                "params": [dist, 0, 0, 0, 0, 0],
                "score": 0.0,
                "sigma": [np.inf] * 5,
                "penalty": 0.0,
                "size": 0,
                "best_idx": 0,
                "is_min": False,
            }
            if self.rank == 0:
                logger.warning(
                    "Skipping Bayesian Optimization because all initial scores are zero."
                )
                logger.warning(
                    "Initial geometry guess is too far from optimum. Please refine the search space."
                )
            return result

        if np.std(y) != 0:
            y_norm = (y - np.mean(y)) / np.std(y)
        else:
            y_norm = y - np.mean(y)

        # 4. Initialize the Gaussian Process model
        kernel = RBF(length_scale=0.3, length_scale_bounds=(0.2, 0.4)) * ConstantKernel(
            constant_value=1.0, constant_value_bounds=(0.5, 1.5)
        ) + WhiteKernel(noise_level=0.001, noise_level_bounds="fixed")
        gp_model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, random_state=0
        )
        gp_model.fit(X_norm_samples, y_norm)
        visited_idx = list([])

        # 5. Run the Bayesian Optimization loop
        for i in range(n_iterations):
            # 6. Select the next point to evaluate
            next = self.UCB(X_norm, gp_model, visited_idx, beta)
            next_sample = X[next]
            visited_idx.append(next)

            # 7. Compute the score of the next point
            score = self.score(next_sample, Imin, max_rings)
            y = np.append(y, [score], axis=0)
            bo_history["params"].append(next_sample)
            bo_history["scores"].append(score)
            X_samples = np.append(X_samples, [X[next]], axis=0)
            X_norm_samples = np.append(X_norm_samples, [X_norm[next]], axis=0)
            if np.std(y) != 0:
                y_norm = (y - np.mean(y)) / np.std(y)
            else:
                y_norm = y - np.mean(y)

            # 8. Update the Gaussian Process model
            gp_model.fit(X_norm_samples, y_norm)

        # 9. Gather results
        best_idx = np.argmax(y)
        best_param = X_samples[best_idx]
        score, sigma, penalty, size, params, is_min = self.gradient_descent(
            best_param, res, Imin, max_rings, step
        )
        logger.info(
            f"Rank {self.rank} dist={dist:.4f}m: score={score:3e}, size={size}, penalty={penalty:3e}"
        )
        result = {
            "bo_history": bo_history,
            "params": params,
            "score": score,
            "sigma": sigma,
            "penalty": penalty,
            "size": size,
            "best_idx": best_idx,
            "is_min": is_min,
        }
        return result

    def bayfai_opt(
        self,
        center,
        bounds,
        res,
        n_samples,
        n_iterations,
        Imin,
        max_rings,
        beta=1.96,
        step=5,
        prior=True,
        seed=None,
    ):
        """
        Run BayFAI optimization.
        Split the distance parameter across MPI ranks.
        Run Bayesian Optimization on each rank with fixed distance.
        Perform pyFAI least-squares refinement for each rank's best geometry.
        Optimal geometry is chosen based on the lowest residual among ranks.

        Parameters
        ----------
        center : dict
            Dictionary of center values for each parameter
        bounds : dict
            Dictionary of bounds for each parameter
        res : dict
            Dictionary of resolution for each parameter
        n_samples : int
            Number of samples to initialize the Gaussian Process
        n_iterations : int
            Number of iterations of Bayesian Optimization
        Imin : float
            Minimum intensity threshold for identifying Bragg peaks
        max_rings : int
            Maximum number of rings to consider
        beta : float
            Exploration-exploitation trade-off parameter for UCB acquisition function
        step : int
            Size of the refinement space around best parameters
        prior : bool
            Whether to sample initial points around the center or randomly
        seed : optional, int
            Random seed for reproducibility
        """
        dist = self.distribute_distances(center, res)
        logger.info(
            f"Rank {self.rank}: Running Bayesian Optimization on distance {dist:.4f} m"
        )

        bayfai_hyperparams = {
            "n_samples": n_samples,
            "n_iterations": n_iterations,
            "Imin": Imin,
            "max_rings": max_rings,
            "beta": beta,
            "step": step,
            "prior": prior,
            "seed": seed,
        }

        results = self.bayes_opt_distance(
            dist,
            center,
            bounds,
            res,
            **bayfai_hyperparams,
        )

        self.comm.Barrier()

        self.scan = {}
        self.scan["bo_history"] = self.comm.gather(results["bo_history"], root=0)
        self.scan["params"] = self.comm.gather(results["params"], root=0)
        self.scan["score"] = self.comm.gather(results["score"], root=0)
        self.scan["size"] = self.comm.gather(results["size"], root=0)
        self.scan["sigma"] = self.comm.gather(results["sigma"], root=0)
        self.scan["penalty"] = self.comm.gather(results["penalty"], root=0)
        self.scan["best_idx"] = self.comm.gather(results["best_idx"], root=0)
        self.scan["is_min"] = self.comm.gather(results["is_min"], root=0)
        self.finalize()

    def finalize(self, lbda=0.2):
        if self.rank == 0:
            for key in self.scan.keys():
                self.scan[key] = np.array([item for item in self.scan[key]])
            self.valid = self.scan["is_min"]
            self.invalid = np.where(~self.valid)[0]
            self.final_score = self.scan["score"] + lbda * self.scan["penalty"]
            self.index = np.argmax(self.final_score)
            self.bo_history = self.scan["bo_history"][self.index]
            self.params = self.scan["params"][self.index]
            self.neglog_score = self.scan["score"][self.index]
            self.size = self.scan["size"][self.index]
            self.sigma = self.scan["sigma"][self.index]
            self.penalty = self.scan["penalty"][self.index]
            self.best_idx = self.scan["best_idx"][self.index]
            self.gr = GeometryRefinement(
                calibrant=self.calibrant,
                dist=self.params[0],
                poni1=self.params[1],
                poni2=self.params[2],
                rot1=self.params[3],
                rot2=self.params[4],
                rot3=self.params[5],
                detector=self.detector,
                wavelength=self.calibrant.wavelength,
            )

    def plot_radial_integration(self, qs, radial, calibrant, ax=None):
        """
        Plot the radial integration of a powder image

        Parameters
        ----------
        qs : np.array
            q-space range covered by detector
        radial : np.array
            Radial intensity profile
        calibrant : Calibrant
            Calibrant object
        ax : plt.Axes
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        unit = RADIAL_UNITS["q_A^-1"]
        ax.plot(qs, radial, color="black", linewidth=0.8)

        x_values = calibrant.get_peaks(unit)
        if x_values is not None:
            for x in x_values:
                line = lines.Line2D(
                    [x, x],
                    ax.axis()[2:4],
                    color="red",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )
                ax.add_line(line)

        ax.set_title("Radial Profile", fontsize=6)
        if unit:
            ax.set_xlabel(unit.label, fontsize=6)
        ax.set_ylabel("Intensity", fontsize=6)
        ax.tick_params(axis="x", labelsize=4)
        ax.tick_params(axis="y", labelsize=4)

    def plot_bo_history(self, ax):
        """
        Plot the Bayesian Optimization history across all ranks

        Parameters
        ----------
        bo_history : list
            List of all the BO histories with keys 'params' and 'scores' for each rank-distance
        ax : plt.Axes
            Matplotlib axes
        """
        bo_history = self.scan["bo_history"]
        iters = np.arange(len(bo_history[self.index]["scores"]))
        ax.plot(
            iters,
            bo_history[self.index]["scores"],
            marker="o",
            markersize=3,
            linestyle="--",
            linewidth=0.8,
            color="black",
            markerfacecolor="red",
            markeredgecolor="black",
            label=f"Best Distance (m): {self.distances[self.index]:.3f}",
        )
        ax.legend(fontsize=6)
        ax.set_xlabel("Iteration", fontsize=6)
        ax.set_ylabel("Score", fontsize=6)
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title("Bayesian Optimization History", fontsize=6)

    def plot_score_distance_scan(self, ax):
        """
        Plot the score scan over distance

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes
        """
        ax.plot(self.distances, self.scan["score"], linewidth=0.8, color="k")
        ax.scatter(
            self.distances[self.valid],
            self.scan["score"][self.valid],
            linewidth=0.8,
            color="green",
            s=10,
        )
        ax.scatter(
            self.distances[self.invalid],
            self.scan["score"][self.invalid],
            linewidth=0.8,
            color="red",
            s=10,
            alpha=0.8,
        )
        ax.set_xlabel("Distance (m)", fontsize=6)
        ax.set_ylabel(
            r"$-\log\left(\frac{1}{N}\sum (2\theta_g - 2\theta_c)^2\right)$", fontsize=6
        )
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(
            "Score vs Distance",
            fontsize=6,
        )

    def plot_residual_distance_scan(self, ax):
        """
        Plot the residual scan over distance

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes
        """
        ax.plot(self.distances, self.final_score, linewidth=0.8, color="k")
        ax.scatter(
            self.distances[self.valid],
            self.final_score[self.valid],
            linewidth=0.8,
            color="green",
            s=10,
        )
        ax.scatter(
            self.distances[self.invalid],
            self.final_score[self.invalid],
            linewidth=0.8,
            color="red",
            marker="x",
            s=10,
        )
        ax.scatter(
            self.distances[self.index],
            self.final_score[self.index],
            color="red",
            s=50,
            marker="*",
        )
        ax.set_xlabel("Distance (m)", fontsize=6)
        ax.set_ylabel(
            r"$-\log\left(\frac{1}{N}\sum (2\theta_g - 2\theta_c)^2\right)$", fontsize=6
        )
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(
            "Penalized Score vs Distance",
            fontsize=6,
        )

    def plot_intensity_hist(self, powder, Imin, ax):
        """
        Plot histogram of pixel intensities in the powder image

        Parameters
        ----------
        powder : np.ndarray
            Powder image
        exp : str
            Experiment name
        run : int
            Run number
        Imin : float
            Minimum intensity threshold for identifying Bragg peaks
        ax : plt.Axes
            Matplotlib axes
        """
        mean = np.mean(powder)
        std_dev = np.std(powder)
        nice_pix = powder[np.where(powder < mean + 2 * std_dev)]
        _ = ax.hist(
            nice_pix.ravel(),
            bins=100,
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
            label="Pixel Intensities",
            orientation="horizontal",
        )
        ax.axhline(
            mean,
            color="red",
            linestyle="--",
            label=f"Mean ({mean:.2f})",
        )
        ax.axhline(
            mean + std_dev,
            color="orange",
            linestyle="--",
            label=f"Mean + Std Dev ({mean + std_dev:.2f})",
        )
        ax.axhline(
            mean + 2 * std_dev,
            color="green",
            linestyle="--",
            label=f"Mean + 2 Std Dev ({mean + 2 * std_dev:.2f})",
        )
        ax.axhline(
            Imin,
            color="purple",
            linestyle=":",
            linewidth=2,
            label=f"Minimum Intensity ({Imin:.2f})",
        )
        ax.set_xlim([0, len(powder.ravel()) // 10])
        ax.set_ylim([0, mean + 2 * std_dev])
        ax.set_ylabel("Pixel Intensity", fontsize=6)
        ax.set_xlabel("Frequency", fontsize=6)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=4)
        ax.set_title(
            f"Histogram of Pixel Intensities \n for {self.exp} run {self.run}",
            fontsize=6,
        )
        ax.legend(fontsize=6)

    def plot_powder_and_resolution(self, ax=None):
        """
        Plot the powder image with calibrated overlapping 2θ rings.

        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib axes
        """
        if ax is None:
            _fig, ax = plt.subplots()
        pixel_index_map = self.detector.pixel_index_map
        y_index = pixel_index_map[..., 0]
        x_index = pixel_index_map[..., 1]

        ax.imshow(
            self.assembled_powder,
            vmin=np.percentile(self.powder, 5),
            vmax=np.percentile(self.powder, 95),
        )
        tth = np.array(self.calibrant.get_2th())
        ttha = calculate_2theta(self.detector, params=self.params)
        for p in range(self.detector.n_modules):
            y = pixel_index_map[p, ..., 0]
            x = pixel_index_map[p, ..., 1]
            ax.contour(
                x,
                y,
                ttha[p],
                levels=tth,
                cmap="autumn",
                linewidths=1,
                linestyles="dashed",
            )

        radii = calculate_radius(self.detector, params=self.params)
        closest_pixel_index = np.argmin(radii)
        closest_pixel = (
            y_index.flatten()[closest_pixel_index],
            x_index.flatten()[closest_pixel_index],
        )
        closest_q = theta2q(
            ttha.flatten()[closest_pixel_index], self.calibrant.wavelength
        )
        closest_resol = 2 * np.pi / closest_q

        furthest_pixel_index = np.argmax(radii)
        furthest_pixel = (
            y_index.flatten()[furthest_pixel_index],
            x_index.flatten()[furthest_pixel_index],
        )
        furthest_q = theta2q(
            ttha.flatten()[furthest_pixel_index], self.calibrant.wavelength
        )
        furthest_resol = 2 * np.pi / furthest_q

        pixel_lvls = np.array([closest_pixel, furthest_pixel])
        resol_lvls = np.array([closest_resol, furthest_resol])
        for pixel, resol in zip(pixel_lvls, resol_lvls):
            ax.text(
                pixel[0],
                pixel[1],
                f"{resol:.3f} \u00c5",
                color="red",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
            )
        ax.set_xlabel("X-axis (m)", fontsize=8)
        ax.set_ylabel("Y-axis (m)", fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(
            f"Run {self.run} - {self.detector.detname} - {self.calibrant_name}",
            fontsize=8,
        )
        ax.set_aspect("equal")

    def create_interactive_powder(
        self,
    ):
        """
        Create an interactive powder image with calibrated overlapping 2θ rings.
        """
        y, x, _ = correct_geom(self.detector)
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        p = figure(
            title=f"Run {self.run} - {self.detector.detname} - {self.calibrant_name}",
            x_axis_label="X-axis (m)",
            y_axis_label="Y-axis (m)",
            width=1200,
            height=1200,
            match_aspect=True,
            x_range=(xmin, xmax),
            y_range=(ymin, ymax),
        )

        vmin, vmax = np.percentile(self.stacked_powder, 5), np.percentile(
            self.stacked_powder, 95
        )
        color_mapper = LinearColorMapper(palette=Viridis256, low=vmin, high=vmax)

        p.image(
            image=[self.assembled_powder[::-1, :]],
            x=xmin,
            y=ymin,
            dw=(xmax - xmin),
            dh=(ymax - ymin),
            color_mapper=color_mapper,
        )

        tth = np.array(self.calibrant.get_2th())
        ttha = calculate_2theta(self.detector, params=self.params)
        for i in range(self.detector.n_modules):
            p.contour(
                x=x[i],
                y=y[i],
                z=ttha[i],
                levels=tth,
                line_color="red",
                line_width=3,
                line_dash="dashed",
            )

        radii = calculate_radius(self.detector, params=self.params)
        closest_pixel_index = np.argmin(radii)
        closest_pixel = (
            x.flatten()[closest_pixel_index],
            y.flatten()[closest_pixel_index],
        )
        closest_q = theta2q(
            ttha.flatten()[closest_pixel_index], self.calibrant.wavelength
        )
        closest_resol = 2 * np.pi / closest_q

        furthest_pixel_index = np.argmax(radii)
        furthest_pixel = (
            x.flatten()[furthest_pixel_index],
            y.flatten()[furthest_pixel_index],
        )
        furthest_q = theta2q(
            ttha.flatten()[furthest_pixel_index], self.calibrant.wavelength
        )
        furthest_resol = 2 * np.pi / furthest_q

        pixel_lvls = np.array([closest_pixel, furthest_pixel])
        resol_lvls = np.array([closest_resol, furthest_resol])
        for pixel, resol in zip(pixel_lvls, resol_lvls):
            label_annotation = Label(
                x=pixel[0],
                y=pixel[1],
                text=f"{resol:.3f} Å",
                text_color="red",
                text_font_size="16pt",
            )
            p.add_layout(label_annotation)

        hover = HoverTool(
            tooltips=[
                ("x", "@x{0.000}"),
                ("y", "@y{0.000}"),
                ("Intensity", "@intensity{0.0}"),
            ]
        )
        p.add_tools(hover)
        p.title.text_font_size = "12pt"
        p.xaxis.axis_label_text_font_size = "10pt"
        p.yaxis.axis_label_text_font_size = "10pt"
        p.xaxis.major_label_text_font_size = "8pt"
        p.yaxis.major_label_text_font_size = "8pt"

        qs = {
            "closest": closest_q,
            "furthest": furthest_q,
        }
        resolutions = {
            "closest": closest_resol,
            "furthest": furthest_resol,
        }
        return p, qs, resolutions

    def create_diagnostics_panel(
        self,
        detector,
        distance,
        plot="",
    ):
        """
        Create a diagnostics panel with the results of the Bayesian Optimization.

        Parameters
        ----------
        detector : PyFAI(Detector)
            Corrected PyFAI detector object
        distance : float
            Refined distance
        plot : str
            Path to save plot
        """
        fig = plt.figure(figsize=(6, 9), dpi=100)
        nrow, ncol = 3, 2
        irow, icol = 0, 0

        # Labelling experiment and run number
        ax1 = plt.subplot2grid((nrow, ncol), (irow, icol))
        rect = patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax1.transAxes,
            color="lightgrey",
            alpha=0.3,
        )
        ax1.add_patch(rect)
        ax1.text(
            0.05,
            0.9,
            f"Experiment {self.exp}",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(0.05, 0.8, f"Run {self.run}", ha="left", va="center", fontsize=8)
        ax1.text(
            0.05,
            0.7,
            f"Detector {self.detector.detname}",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.6,
            f"Calibrant {self.calibrant_name}",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.5,
            f"Distance = {1000 * distance:.3f} ± {1000 * self.sigma[0]:.3f} mm",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.4,
            f"X-shift = {1000 * self.params[1]:.3f} ± {1000 * self.sigma[1]:.3f} mm",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.3,
            f"Y-shift = {1000 * self.params[2]:.3f} ± {1000 * self.sigma[2]:.3f} mm",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.2,
            f"RotX = {self.params[3]:.3f} ± {self.sigma[3]:.6f} rad",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.1,
            f"RotY = {self.params[4]:.3f} ± {self.sigma[4]:.6f} rad",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.axis("off")
        icol += 1

        # Plotting histogram of pixel intensities
        ax2 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_intensity_hist(self.powder, self.Imin, ax2)
        icol = 0
        irow += 1

        # Plotting radial profiles with peaks
        ax3 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=2)
        profile, tth = azimuthal_integration(
            self.powder, detector, params=[distance, 0, 0, 0, 0, 0]
        )
        qs = theta2q(tth, self.calibrant.wavelength)
        self.plot_radial_integration(qs, profile, self.calibrant, ax3)
        irow += 1

        # Plotting score scan over distance
        ax5 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_bo_history(ax5)
        icol += 1

        # Plotting residual scan over distance
        ax6 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_residual_distance_scan(ax6)

        fig.tight_layout()

        if plot != "":
            fig.savefig(plot, dpi=100)
        return fig

    def create_summary_plot(
        self,
        detector,
        distance,
        plot="",
    ):
        """
        Create a summary plot with the results of the Bayesian Optimization.

        Parameters
        ----------
        detector : PyFAI(Detector)
            Corrected PyFAI detector object
        distance : float
            Refined distance
        plot : str
            Path to save plot
        """
        fig = plt.figure(figsize=(9, 12), dpi=100)
        nrow, ncol = 4, 3
        irow, icol = 0, 0

        # Labelling experiment and run number
        ax1 = plt.subplot2grid((nrow, ncol), (irow, icol))
        rect = patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax1.transAxes,
            color="lightgrey",
            alpha=0.3,
        )
        ax1.add_patch(rect)
        ax1.text(
            0.05,
            0.9,
            f"Experiment {self.exp}",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(0.05, 0.8, f"Run {self.run}", ha="left", va="center", fontsize=8)
        ax1.text(
            0.05,
            0.7,
            f"Detector {self.detector.detname}",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.6,
            f"Calibrant {self.calibrant_name}",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.5,
            f"Distance = {1000 * distance:.3f} ± {1000 * self.sigma[0]:.3f} mm",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.4,
            f"ShiftX = {1000 * self.params[1]:.3f} ± {1000 * self.sigma[1]:.3f} mm",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.3,
            f"ShiftY = {1000 * self.params[2]:.3f} ± {1000 * self.sigma[2]:.3f} mm",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.2,
            f"RotX = {self.params[3]:.3f} ± {self.sigma[3]:.6f} rad",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.1,
            f"RotY = {self.params[4]:.3f} ± {self.sigma[4]:.6f} rad",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.axis("off")
        icol += 1

        # Plotting radial profiles with peaks
        ax2 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=ncol - icol)
        profile, tth = azimuthal_integration(
            self.powder, detector, params=[distance, 0, 0, 0, 0, 0]
        )
        qs = theta2q(tth, self.calibrant.wavelength)
        self.plot_radial_integration(qs, profile, self.calibrant, ax=ax2)
        irow += 1
        icol = 0

        # Plotting assembled powder with resolutions
        ax3 = plt.subplot2grid((nrow, ncol), (irow, icol), rowspan=2, colspan=2)
        self.plot_powder_and_resolution(ax=ax3)
        icol = +2

        # Plotting histogram of pixel intensities
        ax4 = plt.subplot2grid((nrow, ncol), (irow, icol), rowspan=2)
        self.plot_intensity_hist(self.powder, self.Imin, ax4)
        irow += 2
        icol = 0

        # Plotting BO convergence
        ax5 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_bo_history(ax5)
        icol += 1

        # Plotting score scan over distance
        ax6 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_score_distance_scan(ax6)
        icol += 1

        # Plotting residual scan over distance
        ax7 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_residual_distance_scan(ax7)

        fig.tight_layout()

        if plot != "":
            fig.savefig(plot, dpi=100)
        return fig
