import svd_util as svderko
import numpy as np


def windowed_noise_filtering(frame_matrix, window_shape, stride=1, threshold_value=0.2, svd_precision=0.1):
    def calculate_elements_to_take(sigmas):
        counter = 0

        while counter < len(sigmas) and sigmas[counter] > threshold_value:
            counter += 1

        if counter == 0:
            counter += 1

        return counter

    def get_svd_with_threshold(windowed_matrix):
        u, sigmas, v_t = svderko.reduced_svd(windowed_matrix, svd_precision)
        elements_to_take = calculate_elements_to_take(sigmas)
        reproduced = svderko.restore_from_reduced(u, sigmas, v_t, elements_to_take)
        return reproduced

    A = np.zeros(frame_matrix.shape)
    occurrences = np.zeros(frame_matrix.shape)
    skip_row = 0
    skip_col = 0

    while skip_row + window_shape[0] <= frame_matrix.shape[0]:
        while skip_col + window_shape[1] <= frame_matrix.shape[1]:
            windowed = frame_matrix[skip_row: skip_row + window_shape[0], skip_col: skip_col + window_shape[1]]
            A[skip_row: skip_row + window_shape[0], skip_col: skip_col + window_shape[1]] += get_svd_with_threshold(
                windowed)

            occurrences[skip_row: skip_row + window_shape[0], skip_col: skip_col + window_shape[1]] += 1

            skip_col += stride

        skip_col = 0
        skip_row += stride

    A /= occurrences

    return A