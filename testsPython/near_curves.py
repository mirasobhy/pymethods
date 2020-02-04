import sys
sys.path.append('..\..')
import pymethods as pma
import pathlib as pt
import numpy as np
path_angio = pt.Path(r'D:\Github\pymethods\testsReconstruction\test_1\angiography')
folder_angio = pma.parse.angiography.Folder(path_angio)

centerline1 = folder_angio.centerline1.data
centerline2 = folder_angio.centerline2.data

point_pairs = centerline1.findPointPairsAtPerpendicularDistance(centerline2, distance=1)

centerline1.plot3d()
centerline2.plot3d()
p1 = point_pairs['on_main']
p2 = point_pairs['on_input']

p1.scatter3d()
p2.scatter3d()

pma.arrays.Vectorspace(np.stack([p1, p2], -1)).plot3d()

pma.pyplot.show()