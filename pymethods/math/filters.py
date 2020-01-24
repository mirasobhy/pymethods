# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:56:34 2019

@author: chris
"""
import numpy as np
import scipy
# python numbers=enable


def sgolay2d(z, window_size, order, derivative=None):
    """sgolay2d
    Apply 2d savitsky golay filter to input array z,
    https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html

    Args:
        z (np.ndarray: array to be smoothed
        window_size (int): size of window, must be odd and greater than the
        order order (int): polynomial order, less than the window size
        derivative (str, optional): get the derivative along a direction, can
        be [None,'col','row','both']. Defaults to None.

    Raises:
        ValueError: 'window_size must be odd'
        ValueError: 'order is too high for the window size'

    Returns:
        np.ndarray: [description]
    """ 
    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    if window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k-n, n) for k in range(order+1) for n in range(k+1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty((window_size**2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros((new_shape))
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = \
        band - np.abs(np.flipud(z[1: half_size + 1, :]) - band)
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] =\
        band + np.abs(np.flipud(z[-half_size-1:-1, :]) - band)
    # left band
    band = np.tile( z[:,0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] =\
        band - np.abs(np.fliplr(z[:, 1:half_size + 1]) - band)
    # right band
    band = np.tile( z[:,-1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = \
        band + np.abs(np.fliplr(z[:, - half_size - 1:-1]) - band)
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] =\
        band - np.abs(
            np.flipud(np.fliplr(z[1:half_size+1, 1:half_size+1])) - band)
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] =\
        band + np.abs(
            np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1])) - band)

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] =\
        band - np.abs(
            np.flipud(Z[half_size+1:2*half_size + 1, -half_size:]) - band)
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] =\
        band - np.abs(
            np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band)

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'),\
            scipy.signal.fftconvolve(Z, -c, mode='valid')


def savgol_filter_cylinder(
        contours, window_size, polyorder, padding=100,
        order=3):
    """savgol_filter_cylinder

    Apply 2d savitsky golay filter to list of contours
    https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html

    Args:
        contours (list[CartContour]): list of contours to be smoothed
        window_size (int)           : size of window, must be odd and greater
                                      than the order
        order (int)                 : polynomial order, less than the window
                                      size
        boundary_correction         : number of contours along the boundary to
                                      fix, i.e not include in smoothing
        padding(int)                : the inlet and outlet need to be manually
                                      padded with padding number of contours
            (smart padding needs to still be implemented)

    Raises:
        ValueError: 'window_size must be odd'
        ValueError: 'order is too high for the window size'

    Returns:
        np.ndarray: filtered contours
    """

    assert padding > window_size
    original_contours = contours.copy()
    contours = np.concatenate(
        [contours[-padding:, :, :],
         contours, contours[0:padding, :, :]])
    inlet = contours[:, None, 0, :]
    outlet = contours[:, None, -1, :]
    boundary_inlet = np.repeat(inlet, padding, axis=1)
    boundary_outlet = np.repeat(outlet, padding, axis=1)
    contours = np.concatenate(
        [boundary_inlet, contours, boundary_outlet],
        axis=1)
    conts = np.arange(contours.shape[1])
    if window_size > order:
        contours = np.stack(
                [sgolay2d(contours[:, :, i], window_size, order) for
                    i in range(contours.shape[-1])], axis=-1)
    contours = contours[padding:-padding, padding:-padding, :]
    return contours
