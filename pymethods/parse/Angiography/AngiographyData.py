import pathlib as pt
import numpy as np
try:
    from pymethods.arrays import Curve
except ImportError:
    from ...arrays import Curve

parsable_methods = {
    'LkebCurve': '_parseLkebCurve',
    'LkebCurveSet': '_parseLkebCurveSet'
}


class Data:
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.lines = f.readlines()
        self.data_class = self._datatypeFromLines(self.lines)
        self.file_path = pt.Path(data_file)
        getattr(
            self, parsable_methods[self.data_class]
        )()

    def _datatypeFromLines(self, lines):
        dataClass = self.lines[0].split(' ')[1]
        parsableClasses = ['LkebCurve', 'LkebCurveSet']
        assert dataClass in parsableClasses, Exception(
            f'{dataClass} not in parsable data types {parsableClasses}')
        return dataClass

    def _parseLkebCurveSet(self):
        self.dataType = self.lines[1].split(' ')[-1].strip('\n')
        self.dimensions = int(self.lines[2].split(' ')[-1])
        self.numberOfCurves = int(self.lines[3].split(' ')[-1])
        self.numberOfPointsPerCurve = []
        self.closed = True
        self.data = []

        dtypeMethod = getattr(np, self.dataType)

        curveStartPoint = 7

        for _ in np.arange(self.numberOfCurves):
            nPoints = int(self.lines[curveStartPoint - 2].split(' ')[-1])
            self.numberOfPointsPerCurve.append(nPoints)
            lines = self.lines[curveStartPoint: (curveStartPoint + nPoints)]
            self.data.append(
                AngiographyData.linesToArray(lines, dtypeMethod=dtypeMethod)
            )
            curveStartPoint += nPoints + 2

        delattr(self, 'lines')

    @classmethod
    def linesToArray(cls, lines, dtypeMethod=np.float, delimeter=' '):
        nPoints = len(lines)
        nDimensions = len(lines[0].split(delimeter))
        array = np.zeros((nPoints, nDimensions))

        for i, line in enumerate(lines):
            array[i] = np.array(
                line.strip('\n').split(delimeter)
            ).astype(dtypeMethod)

        return array


class CenterLineData(Data):

    def __init__(self, dataFile, **kwargs):
        super().__init__(dataFile)
        assert self.dataClass == 'LkebCurve'
        self.data = Curve(self.data, **kwargs)
