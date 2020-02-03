import numpy as np

try:
    from pymethods.arrays import FlatContour, Curve
    from pymethods import math
except ImportError:
    from ...arrays import FlatContour, Curve
    from ... import math

from tqdm import tqdm
import pyvista as pv


class CylindricalSurface(np.ndarray):

    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
            out = np.array(args).view(cls).squeeze()
            assert len(out.shape) == 3
            return out

        if len(args) > 1:
            return np.stack(args, axis=-1)

    def filter(self, **kwargs):
        return self.__class__(
            math.filters.sgolay2d(
                self, padding='periodic_replication', **kwargs)
        )

    def interpolate_long(self, npts, *args, **kwargs):
        newsurface = np.zeros(
            (self.shape[0], self.shape[1], npts)
        )
        for i in range(self.shape[1]):
            line = Curve(self[:, i, :], **kwargs)
            newsurface[:, i, :] = line(
                    np.linspace(0, 1, npts)
                )
        return self.__class__(newsurface)

    def interpolate_contours(self, npts, *args, **kwargs):
        newsurface = np.zeros(
            (self.shape[0], npts, self.shape[-1])
        )
        for i in range(self.shape[-1]):
            line = FlatContour(self[:, :, i], **kwargs)
            newsurface[:, :, i] = line(
                    np.linspace(0, 1, npts)
                )
        return self.__class__(newsurface)

    @classmethod
    def align_contour_points(cls, contour_list, progress=False):
        contour_stack = np.stack(contour_list, axis=-1)
        for i in tqdm(
                np.arange(contour_stack.shape[-1]-1), disable=not progress):
            A = contour_stack[:, 0, i, None]
            B = contour_stack[:, :, i+1]
            distance = math.l2_norm(B-A)
            argmin = np.argmin(distance.squeeze())
            contour_stack[:, :, i+1] = np.roll(B, -argmin, axis=-1)
        return contour_stack

    @classmethod
    def from_contours(cls, contours):
        return cls(cls.align_contour_points(contours))

    def to_vtk(self):
        return pv.StructuredMeshs(*self)


if __name__ == "__main__":
    import sys
    import pymethods as pma
    import pymethods.pyplot as plt
    import pathlib as pt
    import numpy as np
    path_angio = pt.Path(r'F:\GitHub\algorithmsAndStructures\testsReconstruction\test_1\angiography')
    folder_angio = pma.parse.angiography.Folder(path_angio)

    cross_sections = folder_angio.CrossSectionEllipseSet1.data

    surface = CylindricalSurface.from_contours(cross_sections)

    surface = surface.filter()

    print(surface.shape)

    plt.scatter3d(*surface)
    plt.show()

    surface = surface.interpolate_long(200, k=3)
    surface = surface.interpolate_contours(100, k=3)

    plt.scatter3d(*surface)
    plt.show()

    print(surface.shape)
    plt.scatter3d(*surface)
    plt.equal_aspect_3d()
    plt.show()

    surface.to_vtk().show()

