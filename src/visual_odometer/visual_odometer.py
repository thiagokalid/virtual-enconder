import numpy as np
import json
from .displacement_estimators.svd import svd_method
from .preprocessing import image_preprocessing
import threading
class VisualOdometer:
    def __init__(self, img_size,
                 xres=1, yres=1,
                 displacement_algorithm="svd",
                 frequency_window="Stone_et_al_2001",
                 spatial_window="blackman-harris"):

        self.displacement_algorithm = displacement_algorithm
        self.frequency_window = frequency_window
        self.spatial_window = spatial_window
        self.img_size = img_size
        self.xres, self.yres = xres, yres  # Relationship between displacement in pixels and millimeters

        self.current_position = np.array([0, 0])  # In pixels
        self.number_of_displacements = 0

        self.imgs_lock = threading.Lock()
        self.imgs_processed = [None, None]
        # The first img in imgs_processed will always be the last successful image used on a displacement estimation.
        # The second img will be the most recent image

    def calibrate(self, new_xres: float, new_yres: float):
        self.xres, self.yres = new_xres, new_yres

    def save_config(self, path: str, filename="visual-odometer-config"):
        config = {
            "Displacement Algorithm": self.displacement_algorithm,
            "Frequency Window": self.frequency_window,
            "Spatial Window": self.spatial_window,
            "Image Size": self.img_size,
        }
        try:
            with open(path + "/" + filename + ".json", 'w') as fp:
                json.dump(config, fp)
            return True
        except:
            return False

    def estimate_displacement_between(self, img_beg: np.ndarray, img_end: np.ndarray) -> (float, float):
        """
        Estimates the displacement between img_beg and img_end.

        Intendend for single shot usage, for estimating displacements between sequences of images use estimate_last_displacement().
        """
        fft_beg = image_preprocessing(
            img_beg)  # dessa forma é sempre feito o preprocessamento EM DOBRO! (Duas vezes na mesma imagem)
        fft_end = image_preprocessing(img_end)  # dessa forma é sempre feito o preprocessamento EM DOBRO!

        return self._estimate_displacement(fft_beg, fft_end)

    def _estimate_displacement(self, fft_beg, fft_end) -> (float, float):
        if self.displacement_algorithm == "svd":
            _deltax, _deltay = svd_method(fft_beg, fft_end, self.img_size[1], self.img_size[0])  # In pixels
        elif self.displacement_algorithm == "phase-correlation":
            raise NotImplementedError
        else:
            raise TypeError

        # Convert from pixels to millimeters (or equivalent):
        deltax, deltay = _deltax * self.xres, _deltay * self.yres
        self.current_position = np.array([self.current_position[0] + deltax, self.current_position[1] + deltay])
        return deltax * self.xres, deltay * self.yres

    def get_displacement(self):
        try:
            # Compute the displacement:
            img_beg = self.imgs_processed[0]

            with self.imgs_lock:
                img_end = self.imgs_processed[1].copy()
                # Update the image buffer:

            self.imgs_processed[0] = img_end

            displacement = self._estimate_displacement(img_beg, img_end)

            # Update the current position:
            self.current_position[0] += displacement[0]
            self.current_position[1] += displacement[1]

            self.number_of_displacements += 1

            return displacement
        except NotImplementedError:
            return None, None

    def feed_image(self, img: np.ndarray, downsampling_factor: int = 1) -> None:
        # Update the latest image:
        if self.imgs_processed[0] is None:
            # The first iteration
            self.imgs_processed[0] = image_preprocessing(img, downsampling_factor)
        else:
            # Update the current image:
            new_img = image_preprocessing(img, downsampling_factor)
            with self.imgs_lock:
                self.imgs_processed[1] = new_img
