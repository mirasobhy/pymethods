# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.append('..\..')
import pymethods as pma
import pathlib as pt
import numpy as np
path_angio = pt.Path(r'F:\GitHub\algorithmsAndStructures\testsReconstruction\test_1\angiography')
folder_angio = pma.parse.angiography.Folder(path_angio)

contour = folder_angio.BifCoreEllipseSetellipseSet.data[0](np.linspace(0, 1, 100))
centroid = contour.mean(-1)
basis = contour.get_basis()
normal = contour.get_normal()
contour.plot3d()
basis[:, 0].quiver3d(origin=centroid)
basis[:, 1].quiver3d(origin=centroid)
normal.quiver3d(origin=centroid, color='red')
pma.pyplot.equal_aspect_3d()
pma.pyplot.show()