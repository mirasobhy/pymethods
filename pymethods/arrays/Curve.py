try:
    from pymethods import (arrays, math, utils, pyplot)
except ImportError:
    from .. import arrays, pyplot
    from .. import math
    from .. import utils

import numpy as np
from collections import deque
import scipy.interpolate as sci
import scipy.signal as ss
import pyvista as pv


class SplineFunc:

    def __init__(self, splrep):
        self.splrep = splrep

    def __call__(self, s, **kwargs):
        return sci.splev(s, self.splrep, **kwargs)


class Curve(arrays.Vectorspace):

    mode = 'fraction'

    delta = utils.NoInputFunctionAlias('delta_per_point', store=True)
    s = utils.NoInputFunctionAlias('arc_length_per_point', store=True)
    s_tot = utils.NoInputFunctionAlias('total_arc_length', store=True)
    s_frac = utils.NoInputFunctionAlias('fraction_per_point', store=True)

    def __init__(self, *args, **kwargs) -> None:
        reparam_curve = kwargs.pop('reparam_curve', None)
        self._initialize_splineparams(**kwargs)
        self.dim_funcs = deque()
        self._splinify(reparam=reparam_curve)

    def _initialize_splineparams(self, **kwargs):
        self.splineparams = {
            'per': False,
            'k': 3,
            's': None
        }
        self.splineparams.update(
            kwargs
        )

    def delta_per_point(self) -> np.ndarray:
        return np.concatenate(
            [[[0]], math.l2_norm(np.diff(self, axis=-1))], axis=-1
        )

    def arc_length_per_point(self) -> np.ndarray:
        arcdistance = np.zeros(self.shape[-1])
        arcdistance = np.cumsum(self.delta_per_point())
        return arcdistance

    def total_arc_length(self) -> np.ndarray:
        return self.arc_length_per_point()[-1]

    def fraction_per_point(self) -> np.ndarray:
        s = self.arc_length_per_point()
        s_max = self.total_arc_length()
        return s/s_max

    def arc_length_at_fraction(self, fraction) -> np.ndarray:
        return self.total_arc_length() * fraction

    def fraction_at_arc_length(self, arc_length) -> np.ndarray:
        assert np.abs(arc_length) <= self.s_tot
        return arc_length/self.s_tot

    def transport_frames(self):
        return np.array(math.frennet_serret_with_transport(self))

    def findPointPairsAtPerpendicularDistance(
        self, curve, distance=0, resolution=None, tolerance=0.1,
        getClosest=True
    ):
        pointPairs = []
        vtkCurve = pv.Spline(curve.T)
        if resolution is not None:
            interpolatedSelf = self(np.arange(0, 1, resolution))
        else:
            interpolatedSelf = self
        T = interpolatedSelf.transport_frames()[:, :, -1]
        minDistance = 10000
        minID = 10000
        for i, origin in enumerate(interpolatedSelf.T):
            slicedPoints = vtkCurve.slice(
                normal=T[i], origin=origin).points
            if len(slicedPoints) > 0:
                for point in slicedPoints:
                    perpDistance = np.linalg.norm(point - origin)
                    diffFromDesired = np.abs(perpDistance-distance)
                    if diffFromDesired < minDistance:
                        minDistance = diffFromDesired
                        minID = i
                    if diffFromDesired <= tolerance:
                        pointPairs.append(
                            {
                                'on_main': origin,
                                'on_input': point.view(arrays.Vector).astype(
                                    origin.dtype),
                                'error': diffFromDesired
                            }
                        )
        if len(pointPairs) == 0:
            print(
                f' the minimum distance {minDistance} for origin point {minID}')
        if len(pointPairs) != 0:
            if getClosest:
                pointPairs.sort(key=lambda x: x['error'])
                pointPairs = pointPairs[0]
        else:
            pass

        return pointPairs

    def _splinify(self, reparam=None):

        if reparam is None:
            s = self.s_frac
        else:
            s = reparam

        for i in range(self.shape[0]):
            try:
                spline_func = sci.splrep(s, self[i], **self.splineparams)
                self.dim_funcs.append(SplineFunc(spline_func))
            except:
                try:
                    y_unique, unique_inds = np.unique(
                        self[i][0:-2], return_index=True)
                    unique_inds.sort()
                    s_unique = np.concatenate(
                        [s[unique_inds], s[-1, None]])
                    y_unique = np.concatenate(
                        [y_unique, self[i][-1, None]])
                    spline_func = sci.splrep(
                        s_unique, y_unique, **self.splineparams)
                    self.dim_funcs.append(SplineFunc(spline_func))
                except:
                    if len(y_unique) > 1:
                        pass
                    else:
                        raise Exception

    def filter(self, window_length, polyorder, **kwargs):
        self = self.view(np.ndarray)
        return self.__class__(
            ss.savgol_filter(self, window_length, polyorder, **kwargs))

    def __call__(
            self, s, *args, reparam_curve=None, **kwargs) -> np.ndarray:

        if not hasattr(self, 'dim_funcs'):
            self._initialize_splineparams(kwargs)
            self.dim_funcs = deque()
            self._splinify()

        if reparam_curve is None:
            param_curve = self
        else:
            param_curve = Curve(reparam_curve)

        if self.mode in 'arclength':
            s = s/param_curve.s_tot

        assert all([s.max() <= 1, s.min() >= -1])

        if utils.is_iterable(s):
            s[s < 0] = s[s < 0] + 1

        return self.__class__(np.stack([f(s) for f in self.dim_funcs]))

    def to_vtk(self):
        return pv.Spline(self.T)


class Contour(Curve):

    area = utils.NoInputFunctionAlias('calc_area', store=True)
    centroid = utils.NoInputFunctionAlias('calc_centroid', store=True)
    normal = utils.NoInputFunctionAlias('calc_normal', store=True)

    def __new__(cls, *args, **kwargs):
        out = super().__new__(cls, *args, **kwargs)
        return math.close_curve(out).view(cls)

    def _initialize_splineparams(self, **kwargs):
        self.splineparams = {
            'per': True,
            'k': 3,
            's': None
        }
        self.splineparams.update(
            kwargs
        )

    def calc_area(self):
        return math.area(self)

    def calc_centroid(self):
        return arrays.ColumnVector(math.contour_centroid(self))

    def calc_normal(self):
        centered = self - self.centroid
        return arrays.ColumnVector(
            math.approximate_normal(centered)
        )

    def __call__(
            self, s, *args, reparam_curve=None,
            close=True, **kwargs) -> np.ndarray:

        if not hasattr(self, 'dim_funcs'):
            self._initialize_splineparams(kwargs)
            self.dim_funcs = deque()
            self._splinify()

        if reparam_curve is None:
            param_curve = self
        else:
            param_curve = Curve(reparam_curve)

        if self.mode in 'arclength':
            s = s/param_curve.s_tot

        assert all([s.max() <= 1, s.min() >= -1])

        if utils.is_iterable(s):
            s[s < 0] = s[s < 0] + 1
        if close:
            return self.__class__(np.stack([f(s) for f in self.dim_funcs]))
        else:
            return Curve(np.stack([f(s) for f in self.dim_funcs]))

    def filter(
            self, window_length, polyorder, mode='wrap', retries=10, **kwargs):
        import matplotlib.pyplot as plt
        original_class = self.__class__
        self = self.view(np.ndarray)
        signals = []
        rolls = np.linspace(
            len(self.T), retries
        )
        for roll in rolls:
            roll = int(roll)
            rolled = np.roll(self, roll, -1)
            filtered = ss.savgol_filter(
                rolled, window_length, polyorder, mode=mode, **kwargs)
            unrolled = np.roll(filtered, -roll, -1)
            signals.append(unrolled)
        mean = np.mean(signals, 0)
        return original_class(mean)

    def get_basis(self):
        i = math.normalize(self[:, 0, None] - self.centroid).squeeze()
        k = self.get_normal()
        j = math.normalize(math.cross(k, i))
        return arrays.Basis(i, j, k)

    def argsortByBasis(self, basis):
        return math.argSortByBasis(self, basis)

    def sortByBasis(self, basis):
        sorted_args = self.argsortByBasis(basis)
        return self.__class__(self[:, sorted_args.astype(int)])


class FlatContour(Contour):
    """FlatContour
        Flat contours are 3d contours which exist on a plane specified by a
        normal. Note: The contour is automatically converted to 3d
    """
    def __new__(cls, *args, normal=None, **kwargs):

        out = super().__new__(cls, *args, **kwargs)
        centroid = out.centroid
        centered = out - centroid
        if normal is None:
            normal = math.approximate_normal(centered)
        normal = arrays.ColumnVector(normal)
        to_plane = math.project_to_plane(centered, normal)
        return math.close_curve(to_plane+centroid).view(cls)

    def get_normal(self):
        k_test = math.approximate_normal(self-self.centroid)
        i = math.normalize(self[:, 0, None] - self.centroid)
        n = math.normalize(
            self[:, self.shape[-1]//4, None] - self.centroid)
        k_test = math.normalize(
            math.cross(i, n))
        j_test = math.cross(k_test, i)
        alpha1 = math.smallest_angle_between_vectors(j_test, n)
        alpha2 = math.smallest_angle_between_vectors(-j_test, n)
        if alpha1 < alpha2:
            return arrays.Vector(math.normalize(k_test))
        else:
            return arrays.Vector(math.normalize(-k_test))


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')

    if False:
        x = y = z = np.linspace(0, 10, 10)
        curve = Curve(x, y, z)
        interp_curve = curve(
            np.linspace(0, 1, 100)
        )

        ax.scatter(*interp_curve)
        plt.show()

    if False:
        theta = np.linspace(-np.pi, np.pi, 10000)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        contour = Contour(x, y).make_3d() + arrays.ColumnVector(1,2,3)

        normal = contour.normal

        area = contour.area
        true_area = np.pi*r**2

        contour_centroid = contour.centroid()

        contour_interp = contour(
            np.linspace(0, 1, 100)
        )

        ax.scatter(*contour)
        ax.scatter(*contour_interp, color='red')
        ax.scatter(*contour_centroid, color='orange')

        plt.show()

    if True:
        theta = np.linspace(-np.pi, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(x)

        contour = Contour(x,y,z)
        contour_flat = FlatContour(x, y, z, normal=[1,1,-10])
        ax.scatter(*contour)
        ax.scatter(*contour_flat)
        plt.show()