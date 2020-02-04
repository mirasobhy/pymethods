
import sys
sys.path.append('..\..')
import pymethods as pma
import pathlib as pt
import numpy as np
import pymethods.pyplot as plt
path_angio = pt.Path(r'D:\Github\pymethods\testsReconstruction\test_1\angiography')
folder_angio = pma.parse.angiography.Folder(path_angio)

centerline = folder_angio.bifCenterline1.data
contours = folder_angio.CrossSectionEllipseSet1.data

artery_surface = pma.arrays.structured.CylindricalSurface.from_contours(contours)

interp_a = artery_surface.interpolate_long(200)
interp_b = artery_surface.interpolate_long(200, reparam_curve=centerline.s_frac)

interp_b = interp_b.filter(window_size=15)

interp_a.plot3d(color='r')
interp_b.plot3d()
plt.equal_aspect_3d()
plt.show()