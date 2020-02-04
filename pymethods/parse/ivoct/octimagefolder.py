try:
    from pymethods.parse.oct import OCTImage
    from pymethods.utils import enumerate_chunk
except ImportError:
    from . import OCTImage
    from ...utils import enumerate_chunk
import pathlib as pt
import logging
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Queue

logger = logging.getLogger()


def parse_path_name(path, required):
    if required is None:
        return False
    elif path in required:
        return True


class OCTFolder:

    def __init__(self, path):
        self.folder_path = pt.Path(path)
        self.image_files = [
            item for item in self.folder_path.glob('*')
            if any(
                [not ('npy' in item.suffix), not ('npz' in item.suffix)]
            )
        ]

    def save(self, filenmame='parsed_data.npz'):
        np.savez(
            self.folder_path/filenmame,
            contour_data=self.contours,
            landmark_data=self.landmark_data,
            landmark_id=self.landmark_id,
            image_paths=self.image_files,
            landmark_path=self.landmark_path
        )

    def load(self, filename='parsed_data.npz'):
        dataz = np.load(self.folder_path/filename)
        self.contours = dataz['contour_data']
        self.landmark_data = dataz['landmark_data']
        self.landmark_id = dataz['landmark_id']
        self.image_files = dataz['image_paths']
        self.landmark_path = dataz['landmark_path']

    def parse(
            self,
            contourColor=[0.5, 0.3, 0.3], contourCheck=['ge', 'le', 'le'],
            landmarkColor=[0.3, 0.3, 0.5], landmarkCheck=['le', 'le', 'ge'],
            processors=1, landmarkName=None):

        logger.info(
            'Extracting Contours and Landmark Data from Images'
        )
        grabber_args = (
            contourColor, contourCheck, landmarkColor, landmarkCheck)

        if processors == 1:
            contours, landmark = self.grabContourOrLandmarkFromList(
                self.image_files, *grabber_args, landmarkName=landmarkName)
        if processors > 1:
            queue = Queue()
            chunks = list(enumerate_chunk(self.image_files, processors))
            processes = [
                Process(
                    target=OCTFolder._mpGrabContourAndLandmarkFromList,
                    args=(queue, idx, imageList, *grabber_args)
                ) for
                idx, imageList in chunks
            ]
            processed_data = []

            [p.start() for p in processes]
            while 1:
                if not queue.empty():
                    processed_data.append(queue.get())
                running = any(
                    [p.is_alive() for p in processes]
                )
                if not running:
                    break
            [p.join() for p in processes]
            contours, landmark = self._mpPostProcess(processed_data)
            print('done')

        return self._parsePostProcess(contours, landmark)

    def _mpPostProcess(self, mpList):
        mpList.sort(key=lambda x: x[0])
        contours = []
        [contours.extend(a[1]) for a in mpList]
        landmark = [a[-1] for a in mpList if a[-1] is not None][0]
        return contours, landmark

    def _parsePostProcess(self, contours, landmark):
        image_paths = [a['path'] for a in contours]
        contour_data = [a['data'] for a in contours]
        landmark_data = landmark['data']
        landmark_path = landmark['path']
        for i, path in enumerate(image_paths):
            if landmark_path.name in path.name:
                landmark_id = i
        self.contours = contour_data
        self.landmark_data = landmark_data
        self.landmark_id = landmark_id
        self.image_files = image_paths
        self.landmark_path = landmark['path']
        return contour_data, landmark_data, landmark_id

    @staticmethod
    def _mpGrabContourAndLandmarkFromList(queue, id, imageList, *args):
        contours, landmarks = OCTFolder.grabContourOrLandmarkFromList(
            imageList, *args)
        queue.put((id, contours, landmarks))

    @staticmethod
    def grabContourOrLandmarkFromList(
        imageList, contourColor, contourCheck, landmarkColor, landmarkCheck,
        landmarkName=None, **kwargs
    ):
        contours = []
        landmark = None
        N = np.arange(len(imageList))


        for i in tqdm(N, disable=not kwargs.get('progressBar', False)):
            path = pt.Path(imageList[i])
            octImage = OCTImage(path)
            try:
                contour = octImage.get_contour(
                        color=contourColor, check=contourCheck)
                contours.append(
                    {
                        'path': path,
                        'data': contour
                    }
                )
            except:
                logger.debug('%s does not contain a contour' % path.as_posix())
            if any(
                    [landmarkName is None,
                     parse_path_name(path.name, landmarkName)]):
                try:
                    landmarkVector = octImage.get_landmark_vector(
                        color=landmarkColor, check=landmarkCheck).make_3d()
                    contour_centroid = contour.centroid
                    landmarkVector = landmarkVector - contour_centroid
                    landmark = {
                        'path': path,
                        'data': landmarkVector[0:2, :]
                    }
                except:
                    logger.debug(
                        '%s does not contain a landmark' % path.as_posix())
        return contours, landmark


if __name__ == "__main__":
    path = pt.Path(r'D:\Github\pymethods\testsReconstruction\test_1\oct')
    octFolder = OCTFolder(path)
    contours, landmarks, landmark_id = octFolder.parse(processors=1)
    print('done')
