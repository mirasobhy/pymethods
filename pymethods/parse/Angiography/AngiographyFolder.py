import pathlib as pt
try:
    from pymethods.parse.Angiography import Data
except ImportError:
    from .AngiographyData import Data
from abc import abstractmethod
import numpy as np
import os


class Folder:

    parsableClasses = ['LkebCurve', 'LkebCurveSet']

    def __init__(self, folderPath):
        self.folderPath = pt.Path(folderPath)

    @abstractmethod
    def grabParseSave(self):
        NotImplemented

    @abstractmethod
    def quickrun(self):
        NotImplemented

    def getFilepath(self, filename):
        for filePath in self.dataFiles:
            if filename == filePath.name:
                reqFilePath = filePath
                break
        return reqFilePath

    def parseFile(self, filename):
        return Data(self.getFilepath(filename))

    def save(self, filePath):
        np.save(
            filePath,
            self
        )

    @staticmethod
    def fromNP(np_path):
        np_path = pt.Path(np_path)
        if np_path.suffix.lower() == '.npy':
            return np.load(np_path, allow_pickle=True).item()
        if np_path.suffix.lower() == '.npz':
            return np.load(np_path).item()

    @property
    def dataFiles(self): return list(self.folderPath.glob('*.data'))

    @property
    def dataFileNames(self):
        return [filePath.name for filePath in self.dataFiles]

    @property
    def numpyFiles(self):
        files = list(self.folderPath.glob('*'))
        return [
            item for item in files if item.suffix.lower() in ['.npy', '.npz']
        ]

    def clean(self):
        [
            os.remove(file) for file
            in self.folderPath.glob('*')
            if file.name not in angiography_files
        ]


angiography_files = [
    'attribCurve1.data',
    'attribCurve2.data',
    'BifAnalysisInfo.data',
    'bifAttrib1Curve.data',
    'bifAttrib2Curve.data',
    'bifBranch1DiameterCurve.data',
    'bifBranch1StenosisCurve.data',
    'bifBranch2DiameterCurve.data',
    'bifBranch2StenosisCurve.data',
    'bifCenterline1.data',
    'bifCenterline1CorCurve.data',
    'bifCenterline2.data',
    'bifCenterline2CorCurve.data',
    'bifCoreAttribCurve.data',
    'BifCoreEllipseSetellipseSet.data',
    'BifCoreEllipseSetnumPerSlice.data',
    'BifCoreEllipseSetrefEllipseSet.data',
    'BifCoreEllipseSetRefSurface.data',
    'BifCoreEllipseSetstenosisCurve.data',
    'BifCoreEllipseSetSurface.data',
    'bifDistal11AttribCurve.data',
    'bifDistal12AttribCurve.data',
    'BifDistalEllipseSet1ellipseSet.data',
    'BifDistalEllipseSet1numPerSlice.data',
    'BifDistalEllipseSet1refEllipseSet.data',
    'BifDistalEllipseSet1RefSurface.data',
    'BifDistalEllipseSet1stenosisCurve.data',
    'BifDistalEllipseSet1Surface.data',
    'BifDistalEllipseSet2ellipseSet.data',
    'BifDistalEllipseSet2numPerSlice.data',
    'BifDistalEllipseSet2refEllipseSet.data',
    'BifDistalEllipseSet2RefSurface.data',
    'BifDistalEllipseSet2stenosisCurve.data',
    'BifDistalEllipseSet2Surface.data',
    'bifProAttribCurve.data',
    'BifProximalEllipseSetellipseSet.data',
    'BifProximalEllipseSetnumPerSlice.data',
    'BifProximalEllipseSetrefEllipseSet.data',
    'BifProximalEllipseSetRefSurface.data',
    'BifProximalEllipseSetstenosisCurve.data',
    'BifProximalEllipseSetSurface.data',
    'BifurcationAnalysis.txt',
    'centerline1.data',
    'centerline1CorCurve.data',
    'centerline2.data',
    'centerline2CorCurve.data',
    'CrossSectionEllipseSet1.data',
    'CrossSectionEllipseSet2.data',
    'frontalBifContourLeft.data',
    'frontalBifContourMiddle.data',
    'frontalBifContourRight.data',
    'frontalDiameterList1.data',
    'frontalDiameterList2.data',
    'FrontalImage.image',
    'lateralBifContourLeft.data',
    'lateralBifContourMiddle.data',
    'lateralBifContourRight.data',
    'lateralDiameterList1.data',
    'lateralDiameterList2.data',
    'LateralImage.image',
    'refBifCenterline1.data',
    'refBifCenterline2.data',
    'refCenterline1.data',
    'refCenterline2.data',
    'view3D.png'
]
