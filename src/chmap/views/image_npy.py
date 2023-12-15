import numpy as np
from numpy.typing import NDArray

from chmap.views.image import ImageHandler

__all__ = ['NumpyImageHandler']


class NumpyImageHandler(ImageHandler):
    def __init__(self, filename: str, image: NDArray[np.uint] = None, *,
                 resolution: tuple[float, float] = (1, 1)):
        """

        :param filename:
        :param image: Array[uint, [N,], H, W]
        :param resolution: (x, y)
        """
        self.filename = filename
        if image is None:
            image = np.load(self.filename)

        if image.ndim not in (2, 3):
            raise RuntimeError()

        self.image = image
        self.resolution = resolution

    @property
    def __len__(self) -> int:
        if self.image.ndim == 3:
            return len(self.image)
        else:
            return 1

    def __getitem__(self, index: int) -> NDArray[np.uint]:
        pass

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
