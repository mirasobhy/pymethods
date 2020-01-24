try:
    from pymethods import (arrays, math, utils)
except:
    from .. import arrays
    from .. import math
    from .. import utils
    
import scipy.interpolate as scipy
import pyvista as pv
import numpy as np
import typing
from collections import deque
import scipy.interpolate as sci


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
        self._initialize_splineparams()
        # self.splineparams.update(kwargs)
        self.dim_funcs = deque()
        self._splinify()

    def _initialize_splineparams(self):
        self.splineparams = {
            'per': False,
            'k': 3,
            's': None
        }

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
    
    def _splinify(self, reparam=None):

        for i in range(self.shape[0]):
            spline_func = sci.splrep(self.s_frac, self[i], **self.splineparams)
            self.dim_funcs.append(SplineFunc(spline_func))
                        
    def __call__(self, s, *args, reparam_curve=None, **kwargs) -> np.ndarray:
        
        if not hasattr(self, 'dim_funcs'):
            self._initialize_splineparams()
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

class Contour(Curve):
    
    area = utils.NoInputFunctionAlias('calc_area', store=True)
    centroid = utils.NoInputFunctionAlias('calc_centroid', store=True)
    normal = utils.NoInputFunctionAlias('calc_normal', store=True)
    
    def __new__(cls, *args, **kwargs):
        out = super().__new__(cls, *args, **kwargs)
        return math.close_curve(out).view(cls)
    
    def _initialize_splineparams(self):
        self.splineparams = {
            'per': True,
            'k': 3,
            's': None
        }
        
    def calc_area(self):
        return math.area(self)
  
    def calc_centroid(self):
        return arrays.ColumnVector(math.contour_centroid(self))
        
    def calc_normal(self):
        centered = self - self.centroid
        return arrays.ColumnVector(
            math.approximate_normal(centered)
        )
        
class FlatContour(Contour):
    """FlatContour 
        Flat contours are ND contours which exist on a plane specified by a normal
    """    
    def __new__(cls, *args, normal, **kwargs):
        normal = arrays.ColumnVector(normal)
        out = super().__new__(cls, *args, **kwargs)
        centroid = out.centroid
        centered = out - centroid
        to_plane = math.project_to_plane(centered, normal)
        return math.close_curve(to_plane+centroid).view(cls) 

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    
    if False:
        x=y=z=np.linspace(0, 10, 10)
        curve = Curve(x,y,z)
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

    
            
                
            
        