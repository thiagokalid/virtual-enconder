from visual_encoder.displacement_params import DisplacementParams
from visual_encoder.trajectory_estimators import compute_total_trajectory, compute_total_trajectory_path, convert_to_3d
from visual_encoder.phase_correlation import crosspower_spectrum
from visual_encoder.trajectory_estimators import get_img, get_quat_data
from visual_encoder.svd_decomposition import phase_unwrapping, linear_regression
from scipy.sparse.linalg import svds
import numpy as np


# This class contains all parameters required in the global displacement estimation algorithm

class TrajectoryParams(DisplacementParams):
    def __init__(self, disp_params: DisplacementParams, x0=0, y0=0, z0=0):
        super().__init__(method=disp_params.method, spatial_window=disp_params.spatial_window,
                         frequency_window=disp_params.frequency_window)
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.coords = None
        self.n_shots = None

    def get_init_coord(self):
        return self.x0, self.y0

    def get_coords(self, shot=None):
        if shot is not None:
            return self.coords[shot, :]
        else:
            return self.coords

    def compute_cps(self, shot, data_root):
        f = get_img(shot, data_root)
        g = get_img(shot - 1, data_root)
        q, Q = crosspower_spectrum(f, g, self.frequency_window)
        return q, Q

    def compute_svd(self, shot, data_root):
        f = get_img(shot, data_root)
        g = get_img(shot - 1, data_root)
        q, Q = crosspower_spectrum(f, g, method=self.frequency_window)
        qu, s, qv = svds(Q, k=1)
        return qu, s, qv

    def estimate_shift(self, pu_unwrapped, N):
        r = np.arange(0, pu_unwrapped.size)
        M = r.size // 2
        # Choosing a smaller window:
        x = r[M - 50:M + 50]
        y = pu_unwrapped[M - 50:M + 50]
        mu, c = linear_regression(x, y)
        delta = -mu * N / (2 * np.pi)
        return delta, mu, c, x, y

    def set_coords(self, new_coords):
        self.coords = new_coords

    def compute_total_trajectory(self, img_list):
        self.coords = compute_total_trajectory(img_list, self)

    def compute_total_trajectory_path(self, data_root, n_images, n_beg=1, traj_type="2d"):
        coords_2d = compute_total_trajectory_path(data_root, n_images, self, n_beg=n_beg)
        if traj_type == "2d":
            self.coords = coords_2d
        elif traj_type == "3d":
            quaternion_data = get_quat_data(data_root)
            self.coords = convert_to_3d(coords_2d, quaternion_data)
        else:
            pass
        return np.copy(self.coords)

    def calibrate(self, data_root, filename_list, measured_coords, n_images):
        # First element in filename_list is related to X axis calibration, then Y shift calibration.
        resolution = np.zeros(2)  # x and y resolution in pixels / millimeters
        for i, filename in enumerate(filename_list):
            coords = compute_total_trajectory_path(data_root + filename + "/", n_images=n_images[i], traj_params=self)
            resolution[i] = measured_coords[i] / np.abs(coords[-1, 1-i] - coords[0, 1-i])
        return resolution
