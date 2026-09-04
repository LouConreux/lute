"""
Classes for geometry optimization tasks.

Classes:
    BayFAIOpt: optimize LCLS1 detector geometry using PyFAI coupled with Bayesian Optimization.

Functions:
    - _build_ai: Build a pyFAI AzimuthalIntegrator from a detector, geometry parameters, and wavelength.
"""

__all__ = [
    "BayFAIOpt",
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
from bokeh.palettes import Viridis256, Category10  # type: ignore
from bokeh.models.annotations import Label  # type: ignore
import h5py  # type: ignore
import pyFAI  # type: ignore
from pyFAI.geometry import Geometry  # type: ignore
from pyFAI.goniometer import SingleGeometry  # type: ignore
from pyFAI.geometryRefinement import GeometryRefinement  # type: ignore
from pyFAI.calibrant import CALIBRANT_FACTORY  # type: ignore
from pyFAI.units import RADIAL_UNITS  # type: ignore
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator  # type: ignore
from scipy.ndimage import median_filter, zoom  # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # type: ignore
from sklearn.utils._testing import ignore_warnings  # type: ignore
from sklearn.exceptions import ConvergenceWarning  # type: ignore
from mpi4py import MPI

from LCLSGeom.manager import get_geometry  # type: ignore
from LCLSGeom.converter import PsanaToPyFAI, PyFAIToPsana, PyFAIToCrystFEL  # type: ignore

pyFAI.use_opencl = False

logger: logging.Logger = get_logger(__name__)

PHOTON_ENERGY_KEYS: tuple = (
    "ebeamh/ebeamPhotonEnergy",
    "ebeam/ebeamPhotonEnergy",
    "ebeam/photon_energy",
)


def _build_ai(
    detector: pyFAI.detectors.Detector,
    params: list,
    wavelength: float,
) -> AzimuthalIntegrator:
    """
    Build a pyFAI AzimuthalIntegrator from a detector, geometry parameters, and wavelength.

    Parameters
    ----------
    detector : pyFAI.detectors.Detector
        PyFAI detector object.
    params : list
        6 geometry parameters: [dist, poni1, poni2, rot1, rot2, rot3]
    wavelength : float
        X-ray wavelength in meters.

    Returns
    -------
    AzimuthalIntegrator
        Configured pyFAI AzimuthalIntegrator.
    """
    dist, poni1, poni2, rot1, rot2, rot3 = params
    return AzimuthalIntegrator(
        dist=dist,
        poni1=poni1,
        poni2=poni2,
        rot1=rot1,
        rot2=rot2,
        rot3=rot3,
        detector=detector,
        wavelength=wavelength,
    )


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
        exp: str,
        run: int,
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
        detname: str,
        h5: str,
        smooth: bool,
        Imin: float,
        calibrant: str,
        fixed: list,
        wavelength: float = 1e-10,
        median_filter_size: int = 21,
    ):
        """
        Setup the BayFAI optimization.

        Parameters
        ----------
        detname : str
            Name of the detector
        h5 : str
            Path to the smalldata h5 file to use for calibration
        smooth : bool
            If True, remove the background of the powder image
        Imin : float
            Minimum intensity percentile threshold for Bragg peak detection
        calibrant : PyFAI.Calibrant
            PyFAI calibrant object
        fixed : list
            List of parameters to keep fixed during optimization
        wavelength : float, optional
            X-ray wavelength in meters. If provided (non-default 1e-10),
            overrides the value read from the h5 file.
        median_filter_size : int, optional
            Size in pixels of the median filter used to estimate the background.

        Returns
        -------
        Imin : float
            Minimum intensity value for identifying Bragg peaks
        """
        self.detector = self.build_detector(detname)
        self.median_filter_size = median_filter_size if smooth else None
        self.powder = self.generate_powder(
            h5, detname, smooth, Imin, median_filter_size
        )
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
        Imin: float = 95,
        median_filter_size: int = 21,
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
            If True, remove the background of the powder image.
        Imin : float, optional
            Minimum intensity percentile threshold for Bragg peak detection.
        median_filter_size : int, optional
            Size in pixels of the median filter used to estimate the background.
        """
        powder = np.asarray(powder, dtype=np.float64).copy()
        good = mask != 0
        powder[~good] = 0
        powder[powder < 0] = 0
        if smooth:
            for p in range(powder.shape[0]):
                panel = powder[p]
                panel_good = good[p]
                filled = np.where(
                    panel_good,
                    panel,
                    np.median(panel[panel_good]) if panel_good.any() else 0.0,
                )
                background = median_filter(
                    filled, size=median_filter_size, mode="nearest"
                )
                powder[p] = panel - background
            powder[powder < 0] = 0
        powder[~good] = 0
        self.powder = powder
        self.stacked_powder = np.reshape(self.powder, self.detector.shape)
        non_zero_pixels = self.powder[self.powder > 0]
        self.Imin = np.percentile(non_zero_pixels, Imin)
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
        self,
        powder_path: str,
        detname: str,
        smooth: bool = False,
        Imin: float = 95,
        median_filter_size: int = 21,
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
            If True, remove the background of the powder image.
        Imin : float, optional
            Minimum intensity percentile threshold for Bragg peak detection.
        median_filter_size : int, optional
            Size in pixels of the median filter used to estimate the background.
        """
        mask = self.detector.geo.get_pixel_mask(mbits=3)
        mask = np.squeeze(mask, axis=0)
        powder = self.extract_powder(powder_path, detname)
        powder = self.preprocess_powder(powder, mask, smooth, Imin, median_filter_size)
        self.assembled_powder = self.assemble_image(powder)
        return powder

    def build_detector(self, detname: str) -> pyFAI.detectors.Detector:
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
        metrology = get_geometry(detname)
        detector = PsanaToPyFAI.convert(in_file=metrology, detname=detname)
        return detector

    def update_geometry(self, out_file: str) -> pyFAI.detectors.Detector:
        """
        Update the geometry and write a new .poni, .geom and .data file

        Parameters
        ----------
        optimizer : BayesGeomOpt
            Optimizer object
        out_file : str
            Path to the output file
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

    def define_calibrant(
        self,
        calibrant_name: str,
        h5: str,
        wavelength: float = 1e-10,
    ) -> pyFAI.calibrant.Calibrant:
        """
        Define calibrant for optimization with appropriate wavelength.

        Parameters
        ----------
        calibrant_name : str
            Name of the calibrant
        h5 : str
            Path to the smalldata h5 file containing the photon energy.
        wavelength : float, optional
            X-ray wavelength in meters.
        """
        self.calibrant_name = calibrant_name
        calibrant = CALIBRANT_FACTORY(calibrant_name)
        if wavelength != 1e-10:
            logger.info(f"Using user-provided wavelength {wavelength} m")
        else:
            try:
                with h5py.File(h5) as f:
                    for key in PHOTON_ENERGY_KEYS:
                        if key not in f:
                            continue
                        energies = np.asarray(f[key][()], dtype=float)
                        energies = energies[np.isfinite(energies) & (energies > 0)]
                        if energies.size == 0:
                            continue
                        photon_energy = float(np.mean(energies))
                        wavelength = 1.23984193e-6 / photon_energy
                        logger.info(
                            f"Read {photon_energy:.2f} eV from {key} "
                            f"-> wavelength {wavelength:.4e} m"
                        )
                        break
                    else:
                        raise KeyError(
                            f"None of {PHOTON_ENERGY_KEYS} hold usable photon "
                            f"energies in the h5 file"
                        )
            except Exception as e:
                logger.warning(
                    f"Could not read photon energy from {h5} due to {e}, defaulting to provided wavelength {wavelength} m"
                )
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

    def extract_data(self, sample, Imin, max_rings, pts_per_deg):
        """
        Extract data points for a given sampled geometry.

        Parameters
        ----------
        sample : list
            Geometry parameters
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider
        pts_per_deg : int
            Number of points per degree for extraction

        Returns
        -------
        data : np.ndarray
            Extracted data points
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
            "Data Extraction Geometry",
            self.stacked_powder,
            calibrant=self.calibrant,
            detector=self.detector,
            geometry=geom_sample,
        )
        sg.extract_cp(max_rings=max_rings, pts_per_deg=pts_per_deg, Imin=Imin)
        return sg.geometry_refinement.data

    def create_dataset(self, data, bragg_threshold):
        """
        Build a common pixel dataset from all BO ranks.

        Parameters
        ----------
        data : list
            List of control points (d1, d2, ring_index)
        bragg_threshold : int
            Agreement threshold between ranks

        Returns
        -------
        pixels : np.ndarray
            Deduplicated integer pixel positions (d1, d2)
        """
        valid = [d for d in data if d is not None and len(d) > 0]
        if not valid:
            return np.array([], dtype=np.int32).reshape(0, 2)
        all_pixels = np.vstack([np.round(d[:, :2]).astype(np.int32) for d in valid])
        pixels, counts = np.unique(all_pixels, axis=0, return_counts=True)
        return pixels[counts >= bragg_threshold]

    def index_pixels(self, pixels, sample):
        """
        Assign ring indices to a set of pixel positions using a given geometry.

        Parameters
        ----------
        pixels : np.ndarray
            Integer pixel positions (d1, d2)
        sample : array-like of 6 floats
            Geometry parameters [dist, poni1, poni2, rot1, rot2, rot3].

        Returns
        -------
        data : np.ndarray
            Control points (d1, d2, ring_index) with ring labels consistent
            sample geometry.
        """
        dist, poni1, poni2, rot1, rot2, rot3 = sample
        geom = Geometry(
            dist=dist,
            poni1=poni1,
            poni2=poni2,
            rot1=rot1,
            rot2=rot2,
            rot3=rot3,
            detector=self.detector,
            wavelength=self.calibrant.wavelength,
        )

        # Compute 2θ for all pixels in one vectorised call.
        tth_pixels = geom.tth(pixels[:, 0], pixels[:, 1])  # shape (M,)

        # Expected ring positions and tolerance (half the smallest ring gap).
        ring_tth = np.array(self.calibrant.get_2th())
        delta = np.min(np.diff(ring_tth)) / 2

        # Nearest-ring assignment: (n_rings, M) distance matrix.
        dist_matrix = np.abs(ring_tth[:, None] - tth_pixels[None, :])
        ring_indices = np.argmin(dist_matrix, axis=0)  # (M,)
        min_dist = np.min(dist_matrix, axis=0)  # (M,)

        mask = min_dist < delta
        return np.column_stack(
            [pixels[mask].astype(np.float64), ring_indices[mask].astype(np.int32)]
        )

    def score(self, sample, Imin, max_rings, pts_per_deg):
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
        sg.extract_cp(max_rings=max_rings, pts_per_deg=pts_per_deg, Imin=Imin)
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
            penalty = 0.0
            return sigmas, penalty

        cov = np.linalg.inv(hessian)
        sigmas = f_min * np.diag(cov) / dof
        sigmas = np.sqrt(sigmas)
        penalty = -np.log(np.linalg.det(cov)) / 2
        return sigmas, penalty

    def gradient_descent(self, sample, data, resolutions, step=5):
        """
        Run gradient descent refinement on the given sample parameters using pyFAI.

        Parameters
        ----------
        sample : list
            Sample parameters for the refinement
        data : np.ndarray
            Data used for the refinement
        resolutions : dict
            Resolution per parameter for restricted refinement
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
        params : list
            Refined parameters
        """
        dist, poni1, poni2, rot1, rot2, rot3 = sample

        if data is None or len(data) == 0:
            return 0.0, [np.inf] * 5, 0.0, sample

        gr = GeometryRefinement(
            data=data,
            calibrant=self.calibrant,
            dist=dist,
            poni1=poni1,
            poni2=poni2,
            rot1=rot1,
            rot2=rot2,
            rot3=rot3,
            detector=self.detector,
            wavelength=self.calibrant.wavelength,
        )
        gr.set_dist_min(dist - step * resolutions["dist"])
        gr.set_dist_max(dist + step * resolutions["dist"])
        gr.set_poni1_min(poni1 - step * resolutions["poni1"])
        gr.set_poni1_max(poni1 + step * resolutions["poni1"])
        gr.set_poni2_min(poni2 - step * resolutions["poni2"])
        gr.set_poni2_max(poni2 + step * resolutions["poni2"])
        gr.set_rot1_min(rot1 - step * resolutions["rot1"])
        gr.set_rot1_max(rot1 + step * resolutions["rot1"])
        gr.set_rot2_min(rot2 - step * resolutions["rot2"])
        gr.set_rot2_max(rot2 + step * resolutions["rot2"])
        fix = ["rot3", "wavelength"]
        score = -np.log(gr.refine3(fix=fix))
        sigma, penalty = self.estimate_uncertainty(gr)
        params = gr.param
        self.gr = gr
        return score, sigma, penalty, params

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
        pts_per_deg,
        beta=1.96,
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
        pts_per_deg : float
            Number of Bragg peaks to extract per azimuthal degree
        beta : float
            Exploration-exploitation trade-off parameter for UCB acquisition function
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
            y[i] = self.score(X_samples[i], Imin, max_rings, pts_per_deg)
            bo_history["params"].append(X_samples[i])
            bo_history["scores"].append(y[i])

        if np.all(y == 0.0):
            result = {
                "bo_history": bo_history,
                "params": [dist, 0, 0, 0, 0, 0],
                "data": [],
                "score": 0.0,
                "best_idx": 0,
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
            score = self.score(next_sample, Imin, max_rings, pts_per_deg)
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
        best_score = self.score(best_param, Imin, max_rings, pts_per_deg)
        data = self.extract_data(best_param, Imin, max_rings, pts_per_deg)

        logger.info(
            f"Rank {self.rank} dist={dist:.4f}m: score={score:3e}, size={len(data)}"
        )
        result = {
            "bo_history": bo_history,
            "params": best_param,
            "score": best_score,
            "data": data,
            "best_idx": best_idx,
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
        pts_per_deg,
        beta=1.96,
        step=5,
        prior=True,
        seed=None,
        bragg_threshold=2,
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
        pts_per_deg: float
            Number of Bragg peaks to extract per azimuthal degree
        beta : float
            Exploration-exploitation trade-off parameter for UCB acquisition function
        step : int
            Size of the refinement space around best parameters
        prior : bool
            Whether to sample initial points around the center or randomly
        seed : optional, int
            Random seed for reproducibility
        bragg_threshold : int
            Minimum number of BO ranks that must have extracted a pixel for it
            to be included in the common gradient-descent dataset.
        """
        # Distribute distances across MPI ranks
        dist = self.distribute_distances(center, res)
        logger.info(
            f"Rank {self.rank}: Running Bayesian Optimization on distance {dist:.4f} m"
        )

        bayfai_hyperparams = {
            "n_samples": n_samples,
            "n_iterations": n_iterations,
            "Imin": Imin,
            "max_rings": max_rings,
            "pts_per_deg": pts_per_deg,
            "beta": beta,
            "prior": prior,
            "seed": seed,
        }

        # Run BO on the distributed distance for this rank
        results = self.bayes_opt_distance(
            dist,
            center,
            bounds,
            res,
            **bayfai_hyperparams,
        )

        # Gather BO-level results from all ranks
        self.comm.Barrier()
        self.scan = {}
        self.scan["bo_history"] = self.comm.gather(results["bo_history"], root=0)
        self.scan["bo_score"] = self.comm.gather(results["score"], root=0)
        self.scan["best_idx"] = self.comm.gather(results["best_idx"], root=0)

        # Build a common pixel set from all BO ranks
        all_bo_data = self.comm.gather(results["data"], root=0)
        if self.rank == 0:
            common_pixels = self.create_dataset(all_bo_data, bragg_threshold)
            logger.info(
                f"Common dataset: {len(common_pixels)} pixels"
                f" (bragg_threshold={bragg_threshold})"
            )
        else:
            common_pixels = None
        # Broadcast shared pixel positions to all ranks.
        common_pixels = self.comm.bcast(common_pixels, root=0)

        # Index the common pixels using the rank's best BO geometry.
        data = self.index_pixels(common_pixels, results["params"])
        logger.info(f"Rank {self.rank}: {len(data)} control points after ring indexing")

        # Run PyFAI gradient descent on the rank-labelled common dataset
        gd_score, sigma, penalty, params = self.gradient_descent(
            sample=results["params"],
            data=data,
            resolutions=res,
            step=step,
        )
        logger.info(
            f"Rank {self.rank}: gradient descent score={gd_score:.3e},"
            f" n_pts={len(data)}"
        )

        # Gather Gradient Descent results from all ranks
        self.comm.Barrier()
        self.scan["params"] = self.comm.gather(params, root=0)
        self.scan["score"] = self.comm.gather(gd_score, root=0)
        self.scan["sigma"] = self.comm.gather(sigma, root=0)
        self.scan["penalty"] = self.comm.gather(penalty, root=0)

        # Winner selection
        if self.rank == 0:
            for key in self.scan.keys():
                self.scan[key] = np.array([item for item in self.scan[key]])

            self.final_score = self.scan["score"]
            self.index = np.argmax(self.final_score)
            self.bo_history = self.scan["bo_history"][self.index]
            self.params = self.scan["params"][self.index]
            self.neglog_score = self.scan["score"][self.index]
            self.sigma = self.scan["sigma"][self.index]
            self.penalty = self.scan["penalty"][self.index]
            self.best_idx = self.scan["best_idx"][self.index]

            # Re-index the common pixels with the winner's refined params
            winner_data = self.index_pixels(common_pixels, self.params)
            self.gr = GeometryRefinement(
                data=winner_data,
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

    def plot_radial_integration(self, result1d, calibrant, ax=None):
        """
        Plot the radial integration of a powder image

        Parameters
        ----------
        result1d : pyFAI.containers.Integrate1dResult
            Result from AzimuthalIntegrator.integrate1d, containing .radial (q in Å⁻¹) and .intensity.
        calibrant : Calibrant
            Calibrant object
        ax : plt.Axes
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        unit = RADIAL_UNITS["q_A^-1"]
        ax.plot(result1d.radial, result1d.intensity, color="black", linewidth=0.8)

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

    def plot_2d_integration(self, result2d, calibrant, ax=None):
        """
        Plot the 2D azimuthal integration (cake plot) of a powder image.

        Parameters
        ----------
        result2d : pyFAI.containers.Integrate2dResult
            Result from AzimuthalIntegrator.integrate2d, containing .intensity,
            .radial (q in Å⁻¹), and .azimuthal (χ in °).
        calibrant : pyFAI.calibrant.Calibrant
            Calibrant object for Bragg peak overlay.
        ax : plt.Axes, optional
            Matplotlib axes.
        """
        if ax is None:
            _fig, ax = plt.subplots()

        cake = result2d.intensity
        non_zero = cake[cake > 0]
        vmin = np.percentile(non_zero, 5) if len(non_zero) > 0 else 0
        vmax = np.percentile(non_zero, 95) if len(non_zero) > 0 else 1

        ax.imshow(
            cake,
            extent=[
                result2d.radial.min(),
                result2d.radial.max(),
                result2d.azimuthal.min(),
                result2d.azimuthal.max(),
            ],
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )

        unit = RADIAL_UNITS["q_A^-1"]
        peaks = calibrant.get_peaks(unit)
        if peaks is not None:
            for q_peak in peaks:
                ax.axvline(
                    q_peak, color="red", linestyle="--", linewidth=0.8, alpha=0.7
                )

        ax.set_xlabel(unit.label, fontsize=6)
        ax.set_ylabel(r"$\chi$ (°)", fontsize=6)
        ax.tick_params(axis="x", labelsize=4)
        ax.tick_params(axis="y", labelsize=4)
        ax.set_title("2D Azimuthal Integration", fontsize=6)

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
        )
        ax.axvline(
            mean,
            color="red",
            linestyle="--",
            label=f"Mean ({mean:.2f})",
        )
        ax.axvline(
            mean + std_dev,
            color="orange",
            linestyle="--",
            label=f"Mean + Std Dev ({mean + std_dev:.2f})",
        )
        ax.axvline(
            mean + 2 * std_dev,
            color="green",
            linestyle="--",
            label=f"Mean + 2 Std Dev ({mean + 2 * std_dev:.2f})",
        )
        ax.axvline(
            Imin,
            color="purple",
            linestyle=":",
            linewidth=2,
            label=f"Minimum Intensity ({Imin:.2f})",
        )
        ax.set_xlim([0, mean + 2 * std_dev])
        ax.set_xlabel("Pixel Intensity", fontsize=6)
        ax.set_ylabel("Frequency", fontsize=6)
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

        # Physical coordinates powder plot
        p1, p2, _ = self.detector.calc_cartesian_positions()

        ax.imshow(
            self.assembled_powder,
            extent=[p2.min(), p2.max(), p1.min(), p1.max()],
            vmin=np.percentile(self.powder, 5),
            vmax=np.percentile(self.powder, 95),
        )

        # Ring overlay
        ai = _build_ai(self.detector, self.params, self.calibrant.wavelength)
        tth = np.array(self.calibrant.get_2th())
        ttha = ai.twoThetaArray().reshape(self.detector.calib_shape)
        p1_3d = p1.reshape(self.detector.calib_shape)
        p2_3d = p2.reshape(self.detector.calib_shape)
        for m in range(self.detector.n_modules):
            ax.contour(
                p2_3d[m],
                p1_3d[m],
                ttha[m],
                levels=tth,
                cmap="autumn",
                linewidths=1,
                linestyles="dashed",
            )

        # Control points overlay
        if self.gr.data is not None and len(self.gr.data) > 0:
            d1_idx = self.gr.data[:, 0].astype(int)
            d2_idx = self.gr.data[:, 1].astype(int)
            rings = self.gr.data[:, 2].astype(int)
            cp_x = p2[d1_idx, d2_idx]
            cp_y = p1[d1_idx, d2_idx]
            cmap = plt.get_cmap("tab10")
            for ring_id in np.unique(rings):
                mask = rings == ring_id
                ax.scatter(
                    cp_x[mask],
                    cp_y[mask],
                    s=10,
                    color=cmap(ring_id % 10),
                    alpha=0.6,
                    label=f"Ring {ring_id}",
                    zorder=3,
                )
            ax.legend(
                fontsize=5,
                markerscale=2,
                title=f"N={len(self.gr.data)}",
                title_fontsize=5,
            )

        # Resolution overlay
        radii = ai.rArray()
        q_array = ai.qArray() / 10  # nm⁻¹ → Å⁻¹

        d1c, d2c = np.unravel_index(np.argmin(radii), p1.shape)
        closest_resol = 2 * np.pi / q_array[d1c, d2c]
        ax.text(
            p2[d1c, d2c],
            p1[d1c, d2c],
            f"{closest_resol:.3f} Å",
            color="red",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
        )

        d1f, d2f = np.unravel_index(np.argmax(radii), p1.shape)
        furthest_resol = 2 * np.pi / q_array[d1f, d2f]
        ax.text(
            p2[d1f, d2f],
            p1[d1f, d2f],
            f"{furthest_resol:.3f} Å",
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
        p1, p2, _ = self.detector.calc_cartesian_positions()
        # d1 = vertical (bottom→top), d2 = horizontal (left→right)
        y = np.reshape(p1, self.detector.calib_shape)
        x = np.reshape(p2, self.detector.calib_shape)
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

        vmin, vmax = (
            np.percentile(self.stacked_powder, 5),
            np.percentile(self.stacked_powder, 95),
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

        ai = _build_ai(self.detector, self.params, self.calibrant.wavelength)
        tth = np.array(self.calibrant.get_2th())
        ttha = ai.twoThetaArray().reshape(self.detector.calib_shape)
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

        if self.gr.data is not None and len(self.gr.data) > 0:
            d1_idx = self.gr.data[:, 0].astype(int)
            d2_idx = self.gr.data[:, 1].astype(int)
            rings = self.gr.data[:, 2].astype(int)
            cp_x = p2[d1_idx, d2_idx]
            cp_y = p1[d1_idx, d2_idx]
            n_rings = len(np.unique(rings))
            palette = Category10[max(3, min(10, n_rings))]
            for ring_id in np.unique(rings):
                mask = rings == ring_id
                p.circle(
                    cp_x[mask].tolist(),
                    cp_y[mask].tolist(),
                    size=10,
                    color=palette[ring_id % len(palette)],
                    alpha=0.6,
                    legend_label=f"Ring {ring_id}",
                )
            p.legend.title = f"Control points (N={len(self.gr.data)})"
            p.legend.label_text_font_size = "8pt"
            p.legend.click_policy = "hide"

        radii = ai.rArray().reshape(self.detector.calib_shape)
        q_array = ai.qArray().reshape(self.detector.calib_shape) / 10  # nm⁻¹ → Å⁻¹
        closest_pixel_index = np.argmin(radii)
        closest_pixel = (
            x.flatten()[closest_pixel_index],
            y.flatten()[closest_pixel_index],
        )
        closest_q = q_array.flatten()[closest_pixel_index]
        closest_resol = 2 * np.pi / closest_q

        furthest_pixel_index = np.argmax(radii)
        furthest_pixel = (
            x.flatten()[furthest_pixel_index],
            y.flatten()[furthest_pixel_index],
        )
        furthest_q = q_array.flatten()[furthest_pixel_index]
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
        plot="",
    ):
        """
        Create a diagnostics panel with the results of the Bayesian Optimization.

        Parameters
        ----------
        plot : str
            Path to save plot
        """
        fig = plt.figure(figsize=(8, 9), dpi=100)
        nrow, ncol = 3, 2

        # (0,0) Summary text
        ax1 = plt.subplot2grid((nrow, ncol), (0, 0))
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
            f"Distance = {1000 * self.params[0]:.3f} ± {1000 * self.sigma[0]:.3f} mm",
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

        # (0,1) Pixel histogram
        ax2 = plt.subplot2grid((nrow, ncol), (0, 1))
        self.plot_intensity_hist(self.powder, self.Imin, ax2)

        # Compute 1D azimuthal integration via pyFAI
        ai = _build_ai(self.detector, self.params, self.calibrant.wavelength)
        result1d = ai.integrate1d(self.stacked_powder, npt=256, unit="q_A^-1")

        # (1,0) 1D integration, spanning two cols
        ax3 = plt.subplot2grid((nrow, ncol), (1, 0), colspan=2)
        self.plot_radial_integration(result1d, self.calibrant, ax=ax3)

        # (2,0) Score vs distance
        ax4 = plt.subplot2grid((nrow, ncol), (2, 0))
        self.plot_score_distance_scan(ax4)

        # (2,1) Penalized score vs distance
        ax5 = plt.subplot2grid((nrow, ncol), (2, 1))
        self.plot_residual_distance_scan(ax5)

        fig.tight_layout()

        if plot != "":
            fig.savefig(plot, dpi=100)
        return fig

    def create_summary_plot(
        self,
        plot="",
    ):
        """
        Create a summary plot with the results of the Bayesian Optimization.

        Parameters
        ----------
        plot : str
            Path to save plot
        """
        fig = plt.figure(figsize=(9, 12), dpi=100)
        nrow, ncol = 4, 3

        # Labelling experiment and run number
        ax1 = plt.subplot2grid((nrow, ncol), (0, 0))
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
            f"Distance = {1000 * self.params[0]:.3f} ± {1000 * self.sigma[0]:.3f} mm",
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

        # Compute azimuthal integrations via pyFAI
        ai = _build_ai(self.detector, self.params, self.calibrant.wavelength)
        q_max = float(
            (ai.qArray().reshape(self.detector.calib_shape) / 10).max()
        )  # nm⁻¹ → Å⁻¹, detector-limited q maximum
        result1d = ai.integrate1d(
            self.stacked_powder, npt=256, unit="q_A^-1", radial_range=(0, q_max)
        )
        result2d = ai.integrate2d(
            self.stacked_powder,
            npt_rad=256,
            npt_azim=360,
            unit="q_A^-1",
            radial_range=(0, q_max),
        )

        # (0,1) Pixel histogram, spanning two cols
        ax2 = plt.subplot2grid((nrow, ncol), (0, 1), colspan=2)
        self.plot_intensity_hist(self.powder, self.Imin, ax2)

        # (1,0) Powder rings overlay, spanning two cols and two rows
        ax3 = plt.subplot2grid((nrow, ncol), (1, 0), rowspan=2, colspan=2)
        self.plot_powder_and_resolution(ax=ax3)

        # (1,2) Score vs distance
        ax4 = plt.subplot2grid((nrow, ncol), (1, 2))
        self.plot_score_distance_scan(ax4)

        # (2,2) Penalized score vs distance
        ax5 = plt.subplot2grid((nrow, ncol), (2, 2))
        self.plot_residual_distance_scan(ax5)

        # (3,0) 1D integration, spanning two cols
        ax6 = plt.subplot2grid((nrow, ncol), (3, 0), colspan=2)
        self.plot_radial_integration(result1d, self.calibrant, ax=ax6)

        # (3,2) 2D integration
        ax7 = plt.subplot2grid((nrow, ncol), (3, 2))
        self.plot_2d_integration(result2d, self.calibrant, ax7)

        fig.tight_layout()

        if plot != "":
            fig.savefig(plot, dpi=100)
        return fig
