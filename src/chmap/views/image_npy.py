import logging

import numpy as np
from numpy.typing import NDArray

from chmap.views.image import ImageHandler

__all__ = ['NumpyImageHandler']


class NumpyImageHandler(ImageHandler):
    """
    Load image file as numpy array.

    """

    def __init__(self, filename: str, image: NDArray[np.uint] = None, *,
                 logger: str | logging.Logger = None):
        """

        :param filename:
        :param image: Array[uint, [N,], H, W]
        """
        super().__init__(filename, logger=logger)

        if image is not None and image.ndim not in (2, 3):
            raise RuntimeError()

        self._image: NDArray[np.uint] | None = image

    def __len__(self) -> int:
        if (image := self._image) is not None and image.ndim == 3:
            return len(image)
        else:
            return 1

    def __getitem__(self, index: int) -> NDArray[np.uint] | None:
        if (image := self._image) is None:
            return None
        elif image.ndim == 3:
            return image[index]
        else:
            return image

    @property
    def width(self) -> float:
        if (image := self._image) is None:
            return 0

        r = self.resolution[0]
        if image.ndim == 3:
            return image.shape[2] * r
        else:
            return image.shape[1] * r

    @property
    def height(self) -> float:
        if (image := self._image) is None:
            return 0

        r = self.resolution[1]
        if image.ndim == 3:
            return image.shape[1] * r
        else:
            return image.shape[0] * r

    @property
    def image(self) -> NDArray[np.uint] | None:
        return self._image

    def set_image(self, image: NDArray[np.uint] | None):
        self._image = image
