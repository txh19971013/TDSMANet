# author：TXH
# date：2022/11/28 23:17

import os
import numpy as np
import io_func
import time
from tqdm import tqdm


def point2pix_xy(path_image, path_point):
    t1 = time.perf_counter()
    image = io_func.load_from_npy(path_image)
    _, _, t, b = image.shape
    list_roi_filenames = sorted(os.listdir(path_point))
    num_coord = 0
    for point_filename in list_roi_filenames:
        np_roi = io_func.load_from_npy(path_point + point_filename)
        for i in range(np_roi.shape[0]):
            num_coord += 1
    x = np.zeros((num_coord, t, b))
    y = np.zeros((num_coord))
    index = 0
    for class_code, point_filename in enumerate(tqdm(list_roi_filenames, desc='loading points ...')):
        np_roi = io_func.load_from_npy(path_point + point_filename)
        for i in range(np_roi.shape[0]):
            hw = np_roi[i]
            h = hw[0]
            w = hw[1]
            pix_x = image[h, w, ...]
            x[index, ...] = pix_x
            pix_y = np.array(class_code)
            y[index] = pix_y
            index += 1

    t2 = time.perf_counter()
    print('elapsed time: {:.2f}s'.format((t2 - t1)))

    return x, y


def point2patch_xy(path_image, path_point, m):
    t1 = time.perf_counter()
    assert m % 2 == 1, '中心像元块的边长m必须是奇数！'
    dist = m // 2
    image = io_func.load_from_npy(path_image)
    H, W, T, B = image.shape
    list_roi_filenames = sorted(os.listdir(path_point))
    num_coord = 0
    for point_filename in list_roi_filenames:
        np_roi = io_func.load_from_npy(path_point + point_filename)
        for i in range(np_roi.shape[0]):
            num_coord += 1
    x = np.zeros((num_coord, m, m, T, B))
    y = np.zeros((num_coord))
    index = 0
    for class_code, point_filename in enumerate(tqdm(list_roi_filenames, desc='loading image patches ...')):
        np_roi = io_func.load_from_npy(path_point + point_filename)
        for i in range(np_roi.shape[0]):
            hw = np_roi[i]
            h = hw[0]
            w = hw[1]
            try:
                pix_x = image[h - dist: h + dist + 1, w - dist: w + dist + 1, ...]
                x[index, ...] = pix_x
                pix_y = np.array(class_code)
                y[index] = pix_y
                index += 1
            except Exception:
                x = x[:-1]
                y = y[:-1]
                continue

    t2 = time.perf_counter()
    print('elapsed time: {:.2f}s'.format((t2 - t1)))

    return x, y


if __name__ == '__main__':
    x, y = point2pix_xy()
    print(x.shape)
    print((y == 0).sum())
    print((y == 1).sum())
    print((y == 2).sum())
    print((y == 3).sum())
    print((y == 4).sum())
    print((y == 5).sum())
    print((y == 6).sum())
    print((y == 7).sum())
    print((y == 8).sum())
    print((y == 9).sum())
    x, y = point2patch_xy()
    print(x.shape)
    print((y == 0).sum())
    print((y == 1).sum())
    print((y == 2).sum())
    print((y == 3).sum())
    print((y == 4).sum())
    print((y == 5).sum())
    print((y == 6).sum())
    print((y == 7).sum())
    print((y == 8).sum())
    print((y == 9).sum())
