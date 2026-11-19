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

import contextlib
import functools
import gc
import os
from typing import Optional
from unittest import mock
import numpy as np
import numpy.typing as npt
import logging
import h5py  # type: ignore
import pyFAI  # type: ignore
import pyFAI.goniometer  # type: ignore
import pyFAI.massif  # type: ignore
from pyFAI.ext.mathutil import build_qmask  # type: ignore
from pyFAI.geometry import Geometry  # type: ignore
from pyFAI.geometry.core import _geometry  # type: ignore
from pyFAI.goniometer import SingleGeometry  # type: ignore
from pyFAI.geometryRefinement import GeometryRefinement  # type: ignore
from pyFAI.calibrant import CALIBRANT_FACTORY  # type: ignore
from pyFAI.units import RADIAL_UNITS  # type: ignore
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator  # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel  # type: ignore
from sklearn.utils._testing import ignore_warnings  # type: ignore
from sklearn.exceptions import ConvergenceWarning  # type: ignore
from mpi4py import MPI

from LCLSGeom.manager import get_geometry  # type: ignore
from LCLSGeom.converter import PsanaToPyFAI, PyFAIToPsana, PyFAIToCrystFEL  # type: ignore
from LCLSGeom.utils.pyfai_utils import pyfai_to_image_poni  # type: ignore

pyFAI.use_opencl = False


class _CachedKernelNames:
    """Cache, per kernel class, the names sklearn recomputes on every theta access.

    Stock leaf kernels run inspect.signature (get_params) and dir() (hyperparameters)
    on each L-BFGS objective call; only the names are cached, values are always read.
    """

    def get_params(self, deep=True):
        cls = type(self)
        names = cls.__dict__.get("_param_names")
        if names is None:
            names = cls._param_names = tuple(super().get_params(deep=False))
        return {name: getattr(self, name) for name in names}

    @property
    def hyperparameters(self):
        cls = type(self)
        names = cls.__dict__.get("_hp_names")
        if names is None:
            names = cls._hp_names = sorted(
                a for a in dir(cls) if a.startswith("hyperparameter_")
            )
        return [getattr(self, a) for a in names]


class _Matern(_CachedKernelNames, Matern):
    pass


class _ConstantKernel(_CachedKernelNames, ConstantKernel):
    pass


class _WhiteKernel(_CachedKernelNames, WhiteKernel):
    pass


logger: logging.Logger = get_logger(__name__)


def _qmask_first(ttha, tth_min, tth_max, mask, max_rings):
    """build_qmask over the first max_rings+2 2theta intervals only; full call if too few are non-empty."""
    # ponytail: identical because the intervals do not overlap and extract_cp's ring loop breaks before index K.
    n = len(tth_min)
    K = min(n, max_rings + 2)
    if K == n:
        return build_qmask(ttha, tth_min, tth_max, mask)
    qmask, count = build_qmask(ttha, tth_min[:K], tth_max[:K], mask)
    if np.count_nonzero(count) < max_rings:
        return build_qmask(ttha, tth_min, tth_max, mask)
    return qmask, np.concatenate((count, np.zeros(n - K, dtype=count.dtype)))


class _Massif(pyFAI.massif.Massif):
    """Massif whose peaks_from_area scans only the mask's row band (same points, order, RNG draws)."""

    def peaks_from_area(
        self,
        mask,
        Imin=np.finfo(np.float64).min,
        keep=1000,
        dmin=0.0,
        seed=None,
        **kwarg,
    ):
        """Verbatim pyFAI 2024.9 Massif.peaks_from_area except for the all_points and shuffle lines."""
        res = []
        cnt = 0
        dmin2 = dmin * dmin

        # ponytail: same intp values in row-major order as np.where(mask), np.where over the rows that hold points only.
        rows = np.flatnonzero(mask.any(axis=1))
        if rows.size:
            w = np.nonzero(mask[rows[0] : rows[-1] + 1])
            all_points = np.vstack((w[0] + rows[0], w[1])).T
        else:
            all_points = np.vstack(np.where(mask)).T
        # ponytail: legacy RandomState.permutation(n) is arange(n) plus the same Fisher-Yates random_interval(i),
        # i = n-1..1, as shuffle's multi-dim path: same draws, same rows, without the per-row bounce buffer.
        if len(all_points) > 0:
            all_points = all_points[np.random.permutation(len(all_points))]

        if seed:
            seeds = np.array(list(seed))
            if len(seeds) > 0:
                seeds = seeds[np.random.permutation(len(seeds))]
            all_points = np.concatenate((seeds, all_points))

        msg = "[ %3i, %3i ] -> [ %.1f, %.1f ]"
        for idx in all_points:
            out = self.nearest_peak(idx)
            if out is not None:

                pyFAI.massif.logger.debug(msg, idx[1], idx[0], out[1], out[0])
                p0, p1 = int(round(out[0])), int(round(out[1]))
                if mask[p0, p1]:
                    if (self.data[p0, p1] > Imin) and pyFAI.massif.is_far_from_group(
                        out, res, dmin2
                    ):
                        res.append(out)
                        cnt = 0
            if len(res) >= keep or cnt > keep:
                break
            else:
                cnt += 1
        return res


@contextlib.contextmanager
def _fast_extract_cp(max_rings):
    """Patches active around sg.extract_cp(max_rings=...); outputs and global RNG draws are unchanged."""
    with (
        mock.patch.object(
            pyFAI.goniometer,
            "build_qmask",
            functools.partial(_qmask_first, max_rings=max_rings),
        ),
        mock.patch.object(pyFAI.goniometer, "Massif", _Massif),
    ):
        yield


def _row_runs(rows):
    """Contiguous runs of the sorted row indices, as slices in row order."""
    cut = np.flatnonzero(np.diff(rows) > 1)
    starts, ends = np.r_[rows[0], rows[cut + 1]], np.r_[rows[cut], rows[-1]]
    return [slice(int(a), int(b) + 1) for a, b in zip(starts, ends)]


class _SingleGeometry(SingleGeometry):
    """SingleGeometry whose extract_cp does the per-ring mask ops on the ring's row band only."""

    def extract_cp(self, max_rings=None, pts_per_deg=1.0, Imin=0):
        """Verbatim pyFAI 2024.9 SingleGeometry.extract_cp except for the ring-loop mask ops."""
        G = pyFAI.goniometer
        if self.massif is None:
            if self.detector:
                mask = self.detector.dynamic_mask(self.image)
            else:
                mask = None
            self.massif = G.Massif(self.image, mask)

        tth = np.array([i for i in self.calibrant.get_2th() if i is not None])
        tth = np.unique(tth)
        tth_min = np.zeros_like(tth)
        tth_max = np.zeros_like(tth)
        delta = (tth[1:] - tth[:-1]) / 4.0
        tth_max[:-1] = delta
        tth_max[-1] = delta[-1]
        tth_min[1:] = -delta
        tth_min[0] = -delta[0]
        tth_max += tth
        tth_min += tth
        shape = self.image.shape
        ttha = self.geometry_refinement.twoThetaArray(shape)
        chia = self.geometry_refinement.chiArray(shape)
        rings = 0
        cp = G.ControlPoints(calibrant=self.calibrant)
        if max_rings is None:
            max_rings = tth.size

        qmask, count = G.build_qmask(
            ttha, tth_min, tth_max, self.geometry_refinement.detector.mask
        )
        # ponytail: identical because ring i only lives in rows with lo <= i <= hi; outside them stock's
        # qmask == i and mask2 are all False (mask2 kept zero there), and row-major order is preserved.
        mask2 = np.zeros(qmask.shape, dtype=bool)
        hi = qmask.max(axis=1)
        lo = qmask.min(axis=1, initial=count.size, where=qmask >= 0)
        prev = []
        ms = G.marchingsquares.MarchingSquaresMergeImpl(
            ttha, mask=self.geometry_refinement.detector.mask, use_minmax_cache=True
        )
        for i in range(tth.size):
            if rings >= max_rings:
                break
            if count[i]:
                rings += 1
                runs = _row_runs(np.flatnonzero((lo <= i) & (hi >= i)))
                for s in prev:
                    mask2[s] = False
                prev = runs
                masks = [qmask[s] == i for s in runs]
                parts = [self.image[s][m] for s, m in zip(runs, masks)]
                sub_data = parts[0] if len(parts) == 1 else np.concatenate(parts)
                mean = sub_data.mean(dtype=np.float64)
                std = sub_data.std(dtype=np.float64)
                upper_limit = mean + std
                for s, m in zip(runs, masks):
                    np.logical_and(self.image[s] > upper_limit, m, out=mask2[s])
                size2 = sum(int(mask2[s].sum(dtype=int)) for s in runs)
                if size2 < 1000:
                    upper_limit = mean
                    for s, m in zip(runs, masks):
                        np.logical_and(self.image[s] > upper_limit, m, out=mask2[s])
                    size2 = sum(int(mask2[s].sum()) for s in runs)
                # length of the arc:
                points = ms.find_pixels(tth[i])
                seeds = set((i[0], i[1]) for i in points if mask2[i[0], i[1]])
                # max number of points: 360 points for a full circle
                azimuthal = chia[
                    points[:, 0].clip(0, shape[0]), points[:, 1].clip(0, shape[1])
                ]
                nb_deg_azim = np.unique(np.rad2deg(azimuthal).round()).size
                keep = int(nb_deg_azim * pts_per_deg)
                if keep == 0:
                    continue
                dist_min = len(seeds) / 2.0 / keep
                # why 3.0, why not ?

                G.logger.info(
                    "Extracting datapoint for ring %s (2theta = %.2f deg); "
                    + "searching for %i pts out of %i with I>%.1f, dmin=%.1f",
                    i,
                    np.degrees(tth[i]),
                    keep,
                    size2,
                    upper_limit,
                    dist_min,
                )
                res = self.massif.peaks_from_area(
                    mask2, Imin=Imin, keep=keep, dmin=dist_min, seed=seeds, ring=i
                )
                cp.append(res, i)
        self.control_points = cp
        self.geometry_refinement.data = np.asarray(cp.getList(), dtype=np.float64)
        return cp


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

    min_points = 0  # score() returns 0 below this many control points

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
    def UCB(X, gp_model, visited_idx, beta=1.96, n_cand=None):
        """Return the index into X of the best unvisited UCB point, among n_cand random ones if set."""
        idx = np.delete(np.arange(len(X)), visited_idx)
        if n_cand and len(idx) > n_cand:
            idx = np.random.choice(idx, n_cand, replace=False)
        y_pred, y_std = gp_model.predict(X[idx], return_std=True)
        return idx[np.argmax(y_pred + beta * y_std)]

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
        Imin: float,
        calibrant: str,
        fixed: list,
        parallelized: Optional[str] = None,
        wavelength: float = 1e-10,
    ):
        """
        Setup the BayFAI optimization.

        Parameters
        ----------
        detname : str
            Name of the detector
        h5 : str
            Path to the smalldata h5 file to use for calibration
        Imin : float
            Minimum intensity percentile threshold for Bragg peak detection
        calibrant : PyFAI.Calibrant
            PyFAI calibrant object
        fixed : list
            List of parameters to keep fixed during optimization
        parallelized : str, optional
            Name of the parameter to distribute across MPI ranks as a
            sliding window. If None, ranks instead run full BO differentiated
            only by seed.
        wavelength : float, optional
            X-ray wavelength in meters. If provided (non-default 1e-10),
            overrides the value read from the h5 file.

        Returns
        -------
        Imin : float
            Minimum intensity value for identifying Bragg peaks
        """
        self.detector = self.build_detector(detname)
        # twoThetaArray/chiArray recompute full-grid pixel positions (geometry-independent) on every
        # BO evaluation via fromfunction index arrays; pyFAI callers copy (p1 - poni1), never mutate.
        pos, calc = (
            self.detector.calc_cartesian_positions(),
            self.detector.calc_cartesian_positions,
        )
        for a in pos:
            if a is not None:
                a.flags.writeable = False
        self.detector.calc_cartesian_positions = (
            lambda d1=None, d2=None, center=True, use_cython=True: (
                pos
                if center
                and all(
                    d is None or np.shape(d) == self.detector.shape for d in (d1, d2)
                )
                else calc(d1, d2, center, use_cython)
            )
        )
        self.powder = self.generate_powder(h5, detname, Imin)
        self.calibrant = self.define_calibrant(calibrant, h5, wavelength)
        self.set_search_space(parallelized, fixed)
        # pyFAI's poni setters reset engines and call gc.collect() on every geometry build;
        # freezing the post-import/setup heap keeps those collections from rescanning it.
        gc.freeze()

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
        Imin: float = 95,
    ) -> npt.NDArray[np.float64]:
        """
        Preprocess extracted powder for enhancing optimization

        Parameters
        ----------
        powder : npt.NDArray[np.float64]
            Powder image to use for calibration
        mask : npt.NDArray[np.integer]
            Pixel mask to apply to the powder image
        Imin : float, optional
            Minimum intensity percentile threshold for Bragg peak detection.
        """
        powder = np.asarray(powder, dtype=np.float64).copy()
        good = mask != 0
        powder[~good] = 0
        powder[powder < 0] = 0
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
        Imin: float = 95,
    ) -> npt.NDArray[np.float64]:
        """
        Generate a preprocessed powder image from smalldata reduction.

        Parameters
        ----------
        powder_path : str
            Path to the h5 or npy file containing the powder data.
        detname : str
            Name of the detector
        Imin : float, optional
            Minimum intensity percentile threshold for Bragg peak detection.
        """
        mask = self.detector.geo.get_pixel_mask(mbits=3)
        if mask.shape[0] == 1 and mask.ndim > 3:
            mask = np.squeeze(mask, axis=0)
        powder = self.extract_powder(powder_path, detname)
        powder = self.preprocess_powder(powder, mask, Imin)
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

    def update_geometry(self, out_file: str, detname: str) -> pyFAI.detectors.Detector:
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
        poni_file = os.path.join(path, f"r{self.run:0>4}_{detname}.poni")
        self.gr.save(poni_file)
        pyfai_to_image_poni(
            detector=self.detector,
            in_file=poni_file,
            out_file=os.path.join(path, f"r{self.run:0>4}_{detname}_image.poni"),
        )
        PyFAIToPsana.convert(
            in_file=poni_file,
            detector=self.detector,
            out_file=out_file,
        )
        geom_file = os.path.join(path, f"r{self.run:0>4}_{detname}.geom")
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

    def set_search_space(self, parallelized: Optional[str], fixed: list) -> None:
        """
        Define the search space for the free parameters.

        Parameters
        ----------
        parallelized : str, optional
            Name of the parameter to be distributed across MPI ranks as a
            sliding window. Stays part of the free search space.
        fixed : list
            List of parameters to keep fixed during optimization
        """
        if parallelized and parallelized in fixed:
            raise ValueError(
                f"Parameter '{parallelized}' cannot be both parallelized and fixed."
            )
        self.parallelized = parallelized
        self.fixed = fixed
        self.space = []
        self.order = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        for p in self.order:
            if p not in fixed:
                self.space.append(p)

    def distribute_parallelized(self, bounds, res, dist_spacing="geometric"):
        """
        Split the parallelized parameter's absolute bounds into one window per MPI rank.

        dist with dist_spacing 'geometric': rank r gets [b_r, b_r+1] of
        b = low*(high/low)**(arange(size+1)/size), searched on a 2-point grid at the
        window edges. Otherwise ('linear', or any other parallelized parameter):
        size equal windows of width w = (high-low)/size, grid step w.
        Sets res[p] to this rank's window width (refinement range +-step*w).

        Parameters
        ----------
        bounds : dict
            Absolute bounds per parameter, format: {param: (low, high)}
        res : dict
            Resolution per parameter (updated in place for the parallelized one)
        dist_spacing : str
            'geometric' or 'linear' split of bounds['dist'] when dist is parallelized

        Returns
        -------
        value : float, optional
            Center of this rank's window, or None if no parameter is parallelized.
        """
        self.param_values = None
        self.window = None
        if not self.parallelized:
            return None
        p = self.parallelized
        lo, hi = sorted(bounds[p])
        n = self.size
        if p == "dist" and dist_spacing == "geometric":
            b = lo * (hi / lo) ** (np.arange(n + 1) / n)
            lo_k, hi_k = b[self.rank], b[self.rank + 1]
            w = hi_k - lo_k
            value = (lo_k + hi_k) / 2
            self.param_values = (b[:-1] + b[1:]) / 2
            self.window = np.array([lo_k, hi_k])
        else:
            # ponytail: rounding to 9 decimals makes w the decimal of the config (e.g. 0.0058), so
            # the windows reproduce the old center+k*res sweep bit for bit.
            w = round((hi - lo) / n, 9)
            self.param_values = np.round(lo + (np.arange(n) + 0.5) * w, 6)
            value = self.param_values[self.rank]
            self.window = np.arange(value - w / 2, value + w / 2 + w, w)
        res[p] = self.window_width = w
        logger.info(
            f"tiling rank={self.rank} {p}={value:.6f} lo={self.window[0]:.6f} "
            f"hi={self.window[-1]:.6f} w={w:.6f}"
        )
        return value

    def create_search_space(self, value, center, bounds, res):
        """
        Discretize the search space for the free parameters.

        Parameters
        ----------
        value : float, optional
            Window center of the parallelized parameter on this MPI rank
        center : dict
            Center values for each parameter
        bounds : dict
            Absolute bounds for each parameter, format: {param: (low, high)}
        res : dict
            Resolution per parameter

        Returns
        -------
        X : np.ndarray
            Full 6D geometry space (cartesian product)
        X_norm : np.ndarray
            Normalized search space (between-1 and 1)
        """
        if self.parallelized is not None and value is not None:
            center[self.parallelized] = value
        full_params = {}
        search_params = {}
        for p in self.order:
            if p == self.parallelized and value is not None:
                full_params[p] = self.window
                search_params[p] = full_params[p]
            elif p in self.space:
                low, high = bounds[p]
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
            # ponytail: builtin round(), not np.round: c+b - (c-b) differs from 2b in the last bit for
            # most centers (0.28 - 0.26 != 0.02), and the multivariate_normal draws would change.
            widths = [
                (
                    self.window_width
                    if p == self.parallelized
                    else round(float(abs(bounds[p][1] - bounds[p][0])), 12)
                )
                for p in self.space
            ]
            cov = np.diag([(wd / 5) ** 2 for wd in widths])
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

    @functools.cached_property
    def _massif(self):
        """The powder's Massif, built once: peaks_from_area only reads its data, mask and interpolator."""
        return _Massif(
            self.stacked_powder, self.detector.dynamic_mask(self.stacked_powder)
        )

    @functools.cached_property
    def _pos64(self):
        """float64 buffers for p1 - poni1, p2 - poni2 and p3: calc_tth copies other dtypes to float64."""
        p1, p2, p3 = self.detector.calc_cartesian_positions()
        return (
            np.empty(p1.shape),
            np.empty(p2.shape),
            None if p3 is None else p3.astype(np.float64),
        )

    def _fast_maps(self, sg):
        """Serve extract_cp's Massif and 2theta/chi maps from per-rank caches (same cython kernels).

        chiArray is only read at ring-contour pixels, so chi is computed per indexed pixel instead of
        over the full grid; both skip pyFAI's fromfunction index grids. The maps are left alone unless
        the inputs match.
        """
        # Each extract_cp otherwise allocates and frees ~1 GiB on jungfrau16M; on a node whose memory is
        # full of page cache, re-faulting those pages costs more kernel time than the scoring itself.
        sg.massif = self._massif
        gr = sg.geometry_refinement
        if not (
            self.stacked_powder.shape == self.detector.shape
            and gr._parallax is None
            and gr.chiDiscAtPi is True
            and gr.detector is self.detector
        ):
            return
        p1, p2, p3 = self.detector.calc_cartesian_positions()
        b1, b2, b3 = self._pos64
        # dtype=p1.dtype: the difference is still taken in the positions' precision, then widened.
        gr.twoThetaArray = lambda shape=None: _geometry.calc_tth(
            L=gr._dist,
            rot1=gr._rot1,
            rot2=gr._rot2,
            rot3=gr._rot3,
            pos1=np.subtract(p1, gr._poni1, out=b1, dtype=p1.dtype),
            pos2=np.subtract(p2, gr._poni2, out=b2, dtype=p2.dtype),
            pos3=b3,
        )

        class ChiAt:
            def __getitem__(_, idx):
                return _geometry.calc_chi(
                    L=gr._dist,
                    rot1=gr._rot1,
                    rot2=gr._rot2,
                    rot3=gr._rot3,
                    pos1=p1[idx] - gr._poni1,
                    pos2=p2[idx] - gr._poni2,
                    pos3=None if p3 is None else p3[idx],
                    orientation=gr.detector.orientation,
                    chi_discontinuity_at_pi=gr.chiDiscAtPi,
                )

        gr.chiArray = lambda shape=None: ChiAt()

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
        sg = _SingleGeometry(
            "Data Extraction Geometry",
            self.stacked_powder,
            calibrant=self.calibrant,
            detector=self.detector,
            geometry=geom_sample,
        )
        self._fast_maps(sg)
        with _fast_extract_cp(max_rings):
            sg.extract_cp(max_rings=max_rings, pts_per_deg=pts_per_deg, Imin=Imin)
        return sg.geometry_refinement.data

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
        sg = _SingleGeometry(
            "Score Geometry",
            self.stacked_powder,
            calibrant=self.calibrant,
            detector=self.detector,
            geometry=geom_sample,
        )
        self._fast_maps(sg)
        with _fast_extract_cp(max_rings):
            sg.extract_cp(max_rings=max_rings, pts_per_deg=pts_per_deg, Imin=Imin)
        data = sg.geometry_refinement.data

        # A handful of points can fit any geometry, so too few points is a failed geometry.
        if data is None or len(data) == 0 or len(data) < self.min_points:
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
        penalty = -np.log(np.linalg.det(cov))
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

        if data is None or len(data) == 0 or len(data) < self.min_points:
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
    def bayes_opt_rank(
        self,
        value,
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
        step=5,
        seed=None,
        n_cand=None,
        gp_restarts=10,
    ):
        """
        Run Bayesian Optimization on this MPI rank's search space.

        Parameters
        ----------
        value : float, optional
            Shifted center for the parallelized parameter on this MPI rank,
            or None if no parameter is parallelized.
        center : dict
            Dictionary of center values for each parameter
        bounds : dict
            Dictionary of absolute (low, high) bounds for each parameter
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
        step : int
            Step size for the gradient descent refinement
        seed : optional, int
            Random seed for reproducibility. Offset by this rank's index so
            that every rank is independently seeded.
        gp_restarts : int
            n_restarts_optimizer of the GP kernel-hyperparameter fit
        """
        if seed is not None:
            np.random.seed(seed + self.rank)

        # 1. Create the search space
        X, X_norm = self.create_search_space(value, center, bounds, res)
        center[self.parallelized] = value

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
                "params": np.array(
                    [
                        center[k]
                        for k in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")
                    ]
                ),
                "score": 0.0,
                "sigma": [np.inf] * 5,
                "penalty": 0.0,
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
        kernel = _Matern(
            length_scale=0.3, length_scale_bounds=(0.2, 0.4), nu=2.5
        ) * _ConstantKernel(
            constant_value=1.0, constant_value_bounds=(0.5, 1.5)
        ) + _WhiteKernel(
            noise_level=0.001, noise_level_bounds="fixed"
        )
        gp_model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=gp_restarts, random_state=0
        )
        gp_model.fit(X_norm_samples, y_norm)
        visited_idx = list([])

        # 5. Run the Bayesian Optimization loop
        for i in range(n_iterations):
            # 6. Select the next point to evaluate
            next = self.UCB(X_norm, gp_model, visited_idx, beta, n_cand)
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
            if i < n_iterations - 1:  # the final refit is never read
                gp_model.fit(X_norm_samples, y_norm)

        # 9. Gather results
        best_idx = np.argmax(y)
        best_param = X_samples[best_idx]
        data = self.extract_data(best_param, Imin, max_rings, pts_per_deg)
        score, sigma, penalty, params = self.gradient_descent(
            best_param, data, res, step
        )
        logger.info(f"Rank {self.rank}: score={score:3e}, penalty={penalty:3e}")
        result = {
            "bo_history": bo_history,
            "params": params,
            "score": score,
            "sigma": sigma,
            "penalty": penalty,
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
        prior=True,
        step=5,
        lbda=0.1,
        seed=None,
        n_cand=None,
        dist_spacing="geometric",
        gp_restarts=10,
        min_points=None,
    ):
        """
        Run BayFAI optimization.
        Distribute BO across MPI ranks for the parallelized parameter.
        Run Bayesian Optimization on each rank with fixed search space.
        Perform pyFAI least-squares refinement for each rank's best geometry.
        Optimal geometry is chosen based on the lowest residual among ranks.

        Parameters
        ----------
        center : dict
            Dictionary of center values for each parameter
        bounds : dict
            Dictionary of absolute (low, high) bounds for each parameter
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
        n_cand : optional, int
            Number of random grid candidates scored by UCB per step (None/0 = whole grid)
        dist_spacing : str
            'geometric' (default) or 'linear' split of bounds['dist'] across ranks when
            parallelized == 'dist'
        gp_restarts : int
            n_restarts_optimizer of the GP kernel-hyperparameter fit
        min_points : int, optional
            Geometries with fewer extracted control points score 0 (0 = off; None = 100
            when parallelized == 'dist', else 0)
        """
        # Distance sweeps reach geometries where a few stray points fit; MEC boxes do not.
        if min_points is None:
            min_points = 100 if self.parallelized == "dist" else 0
        self.min_points = min_points
        # Split the parallelized parameter's absolute bounds into one window per MPI rank
        res = dict(res)
        value = self.distribute_parallelized(bounds, res, dist_spacing)
        if self.rank == 0:
            if self.parallelized is not None:
                logger.info(
                    f"Running BayFAI parallelizing {self.size} ranks over '{self.parallelized}'"
                )
            else:
                logger.info(f"Running BayFAI on {self.size} ranks")

        bayfai_hyperparams = {
            "n_samples": n_samples,
            "n_iterations": n_iterations,
            "Imin": Imin,
            "max_rings": max_rings,
            "pts_per_deg": pts_per_deg,
            "beta": beta,
            "step": step,
            "prior": prior,
            "seed": seed,
            "n_cand": n_cand,
            "gp_restarts": gp_restarts,
        }

        # Run BO on this rank's search space
        results = self.bayes_opt_rank(
            value,
            center,
            bounds,
            res,
            **bayfai_hyperparams,
        )

        if self.rank == 0:
            # Rank 0 mostly waits at the Barrier for slower ranks: hide its plotting imports there.
            # They leave numpy's global RNG state untouched, so later pyFAI draws are unchanged.
            import matplotlib.pyplot  # type: ignore # noqa: F401
            import bokeh.plotting  # type: ignore # noqa: F401
            import panel  # type: ignore # noqa: F401

        # Gather BayFAI results from all ranks
        self.comm.Barrier()
        self.scan = {}
        self.scan["bo_history"] = self.comm.gather(results["bo_history"], root=0)
        self.scan["params"] = self.comm.gather(results["params"], root=0)
        self.scan["score"] = self.comm.gather(results["score"], root=0)
        self.scan["sigma"] = self.comm.gather(results["sigma"], root=0)
        self.scan["penalty"] = self.comm.gather(results["penalty"], root=0)
        self.scan["best_idx"] = self.comm.gather(results["best_idx"], root=0)

        # Winner selection
        if self.rank == 0:
            for key in self.scan.keys():
                self.scan[key] = np.array([item for item in self.scan[key]])

            self.final_score = self.scan["score"] + lbda * self.scan["penalty"]
            self.index = np.argmax(self.final_score)
            self.bo_history = self.scan["bo_history"][self.index]
            self.params = self.scan["params"][self.index]
            self.neglog_score = self.scan["score"][self.index]
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
            # One integrator for all rank-0 plots: its 2theta/q arrays and integration engines are cached.
            self.ai = _build_ai(self.detector, self.params, self.calibrant.wavelength)

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
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib import lines  # type: ignore

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
        import matplotlib.pyplot as plt  # type: ignore

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
        if self.param_values is not None:
            label = f"Best {self.parallelized}: {self.param_values[self.index]:.3f}"
        else:
            label = f"Best rank: {self.index}"
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
            label=label,
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
        x = self.param_values if self.param_values is not None else np.arange(self.size)
        xlabel = self.parallelized if self.parallelized is not None else "Rank"
        ax.plot(x, self.scan["score"], linewidth=0.8, color="k")
        ax.set_xlabel(xlabel, fontsize=6)
        ax.set_ylabel(
            r"$-\log\left(\frac{1}{N}\sum (2\theta_g - 2\theta_c)^2\right)$", fontsize=6
        )
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(
            f"Score vs {xlabel}",
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
        x = self.param_values if self.param_values is not None else np.arange(self.size)
        xlabel = self.parallelized if self.parallelized is not None else "Rank"
        ax.plot(x, self.final_score, linewidth=0.8, color="k")
        ax.scatter(
            x[self.index],
            self.final_score[self.index],
            color="red",
            s=50,
            marker="*",
        )
        ax.set_xlabel(xlabel, fontsize=6)
        ax.set_ylabel(
            r"$-\log\left(\frac{1}{N}\sum (2\theta_g - 2\theta_c)^2\right)$", fontsize=6
        )
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(
            f"Penalized Score vs {xlabel}",
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
        import matplotlib.pyplot as plt  # type: ignore

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
        ai = self.ai
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
        from bokeh.plotting import figure  # type: ignore
        from bokeh.models import LinearColorMapper, HoverTool  # type: ignore
        from bokeh.palettes import Viridis256, Category10  # type: ignore
        from bokeh.models.annotations import Label  # type: ignore

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

        ai = self.ai
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
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.patches as patches  # type: ignore

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
        ai = self.ai
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
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.patches as patches  # type: ignore

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
        ai = self.ai
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
