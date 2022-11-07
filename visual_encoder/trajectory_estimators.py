import numpy as np
from visual_encoder.dip_utils import gaussian_noise, salt_and_pepper, apply_window
from visual_encoder.phase_correlation import pc_method
from visual_encoder.svd_decomposition import svd_method

def compute_new_position(deltax, deltay, x0, y0):
    xf = x0 + deltax
    yf = y0 + deltay
    return xf, yf


def compute_trajectory(img_beg, img_end, x0, y0, method='pc', window_type=None):
    # Aplica algum tipo de janelamento para mitigar os edging effects:
    img_beg, img_end = apply_window(img_beg, img_end, window_type)

    if method == 'pc':
        deltax, deltay = pc_method(img_beg, img_end)
    elif method == 'svd':
        deltax, deltay = svd_method(img_beg, img_end)
    else:
        raise ValueError('Selected method not supported.')
    xf, yf = compute_new_position(deltax, deltay, x0, y0)
    return xf, yf


def compute_total_trajectory(img_list, x0, y0, method='pc', window_type=None):
    positions = np.zeros((len(img_list), 2))
    positions[0, :] = (x0, y0)
    for i in range(1, len(img_list)):
        if i == 69:
            pass
        positions[i, :] = compute_trajectory(img_list[i - 1], img_list[i], x0, y0, method=method, window_type=window_type)
        x0, y0 = positions[i, :]
    return positions


def generate_artifical_shifts(base_image, width=None, height=None, x0=0, y0=0, xshifts=None, yshifts=None, steps=100,
                              gaussian_noise_db=None, salt_pepper_noise_prob=None):
    if xshifts is None or yshifts is None:
        # Trajetória em diagonal
        xshifts = np.zeros(steps)
        yshifts = np.zeros(steps)
        xshifts[:steps // 4] = 4
        yshifts[:steps // 4] = 0
        xshifts[steps // 4:2 * steps // 4] = 0
        yshifts[steps // 4:2 * steps // 4] = 4
        xshifts[2 * steps // 4:3 * steps // 4] = -4
        yshifts[2 * steps // 4:3 * steps // 4] = 0
        xshifts[3 * steps // 4:4 * steps // 4] = 0
        yshifts[3 * steps // 4:4 * steps // 4] = -4

    xshifts[0] = 0
    yshifts[0] = 0
    if width is None or height is None:
        width = int(base_image.shape[0] / 4)
        height = int(base_image.shape[1] / 4)
    img_list = list()
    coordinates = np.zeros((steps, 2))
    for i in range(steps):
        coordinates[i, :] = (x0, y0)
        y0 = int(y0 + yshifts[i])
        x0 = int(x0 + xshifts[i])
        yf = int(y0 + height)
        xf = int(x0 + width)
        shifted_img = base_image[y0:yf, x0:xf]
        if gaussian_noise_db is not None:
            shifted_img = gaussian_noise(shifted_img, gaussian_noise_db)
        if salt_pepper_noise_prob is not None:
            shifted_img = salt_and_pepper(shifted_img, prob=salt_pepper_noise_prob)
            # plt.imshow(shifted_img)

        img_list.append(shifted_img)
    salt_and_pepper(shifted_img)
    return img_list, xshifts, yshifts, coordinates
