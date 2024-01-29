import numpy as np
from numpy.typing import NDArray

from chmap.views.image import ImageHandler

__all__ = ['NumpyImageHandler']


class NumpyImageHandler(ImageHandler):
    def __init__(self, filename: str, image: NDArray[np.uint] = None):
        """

        :param filename:
        :param image: Array[uint, [N,], H, W]
        """
        self.filename = filename
        if image is None:
            image = np.load(self.filename)

        if image.ndim not in (2, 3):
            raise RuntimeError()

        self.image = image
        self._resolution = (1, 1)

    def __len__(self) -> int:
        if self.image.ndim == 3:
            return len(self.image)
        else:
            return 1

    def __getitem__(self, index: int) -> NDArray[np.uint]:
        if self.image.ndim == 3:
            return self.image[index]
        else:
            return self.image

    @property
    def resolution(self) -> tuple[float, float]:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: float | tuple[float, float]):
        if not isinstance(resolution, tuple):
            resolution = float(resolution)
            resolution = (resolution, resolution)
        self._resolution = resolution

    @property
    def width(self) -> float:
        r = self.resolution[0]
        if self.image.ndim == 3:
            return self.image.shape[2] * r
        else:
            return self.image.shape[1] * r

    @property
    def height(self) -> float:
        r = self.resolution[1]
        if self.image.ndim == 3:
            return self.image.shape[1] * r
        else:
            return self.image.shape[0] * r
