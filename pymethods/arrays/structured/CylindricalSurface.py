import numpy as np
import logging
try:
    from pymethods.arrays import FlatContour, Curve
    from pymethods import math
except ImportError:
    from ...arrays import FlatContour, Curve
    from ... import math

from tqdm import tqdm
import pyvista as pv

logger = logging.getLogger().setLevel(logging.INFO)


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

    def align_contour_points(self, progress=False):
        if isinstance(self, (list, tuple)):
            contour_stack = np.stack(self, axis=-1)
        else:
            contour_stack = self

        converged = False
        shortest_length = np.inf

        while not converged:

            for i in tqdm(
                    np.arange(contour_stack.shape[-1]-1), disable=not progress):
                A = contour_stack[:, 0, i, None]
                B = contour_stack[:, :, i+1]
                distance = math.l2_norm(B-A)
                argmin = np.argmin(distance.squeeze())
                contour_stack[:, :, i+1] = np.roll(B, -argmin, axis=-1)

            # find the shortest path
            deltas = contour_stack[:, :, 1:] - contour_stack[:, :, 0:-1]
            deltas = np.linalg.norm(deltas, axis=0)
            all_lengths = deltas.sum(-1)
            i_short = np.argmin(all_lengths)
            current_shortest = all_lengths[i_short]
            if shortest_length > current_shortest:
                shortest_length = current_shortest
            else:
                break
            contour_stack = np.roll(contour_stack, -i_short, axis=1)
            logging.debug(
                'rollval=%d, shortest_length=%0.3f' % (
                    i_short, shortest_length))

        # now find the shortest path and make that into the cutline
        return CylindricalSurface(contour_stack)

    @classmethod
    def from_contours(cls, contours):
        return cls(cls.align_contour_points(contours))

    def to_vtk(self):
        return pv.StructuredGrid(*self)


if __name__ == "__main__":
    import sys
    import pymethods as pma
    import pymethods.pyplot as plt
    import pathlib as pt
    import numpy as np
    path_angio = pt.Path(r'D:\Github\algorithmsAndStructures\testsReconstruction\test_1\angiography')
    folder_angio = pma.parse.angiography.Folder(path_angio)

    cross_sections = folder_angio.CrossSectionEllipseSet1.data

    surface = CylindricalSurface.from_contours(cross_sections)

    surface = surface.filter()

    plt.scatter3d(
        *surface
    )

    plt.show()

    surface = surface.interpolate_long(720, k=3)
    surface = surface.interpolate_contours(360, k=3)
    surface = surface.align_contour_points()
    surface = surface.filter(window_size=13)
    surface = surface.filter(window_size=15)
    surface = surface.filter(window_size=43)

    surface.to_vtk().plot()

