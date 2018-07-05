#!/usr/bin/env python3
"""
initial_func.py - Packages for the wave equation numerical solver. These
equations act as the initial position and velocity functions for the 1D and
2D problems.

11May18 - Abel Flores Prieto & Sang Hun Chou
"""

import numpy as np


# -------- For 1D Waves ----------------

def pulse_trian1d(x, L):
    """Triangular pulse centered in a string of length L"""
    array = []
    for i in x:
        if i >= L / 2 - L / 10 and i <= L / 2:
            array.append(i - (L / 2 - L / 10))
        elif i >= L / 2 and i <= L / 2 + L / 10:
            array.append(-i + (L / 2 + L / 10))
        else:
            array.append(0)
    return np.array(array)


def pulse_sine1d(x, L):
    """Sine wave pulse centered in a string of length L"""
    array = []
    alpha = 2 * np.pi / (L / 2 - L / 10)
    for i in x:
        if i >= L / 2 - L / 10 and i <= L / 2 + L / 10:
            array.append(np.sin(alpha * i))
        else:
            array.append(0)
    return np.array(array)


def pulse_square1d(x, L):
    """Square wave pulse contered in a string of length L"""
    array = []
    for i in x:
        if i >= L / 2 - L / 10 and i <= L / 2 + L / 10:
            array.append(1)
        else:
            array.append(0)
    return np.array(array)


def init_vel1d(x):
    """Initial wave velocity. Almost always set to 0 for our puposes."""
    return np.zeros(x.shape)


# --------- 2D Waves (Cartesian) --------------


def wave_gaussian2d(x, y, Lx, Ly):
    """
    Gaussian wave centered in sheet of length Lx by Ly.
    f_init = e^(-((x - Lx/2)^2 + (y - Ly/2)) / 0.01) / 0.01
    """
    y, x = np.meshgrid(y, x)
    mu = 0.1
    f = np.exp(-((x - 0.5 * Lx)**2 + (y - 0.5 * Ly)**2) / mu**2) / mu**2
    return f


def wave_polynomial2d(x, y, Lx, Ly):
    """
    Polynomial wave in 2D.
    f_init = 4x^2*y*(1-x)*(1-y)
    """
    y, x = np.meshgrid(y, x)
    f = 4 * x**2 * y * (1 - x) * (1 - y)
    return f


def wave_trig2d(x, y, Lx, Ly):
    """
    Standing sine and cosine wave.
    f_init = sin(2pi*x)*cos(2pi*y)
    """
    y, x = np.meshgrid(y, x)
    f = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    return f


def wave_trig_gauss2d(x, y, Lx, Ly):
    """
    Trigonometric times gaussian wave.
    f_init = sin(2pi*x)*cos(2pi*y)*e^(-((x-Lx/2)^2 + (y-Ly/2)) / 0.01) / 0.01
    """
    y, x = np.meshgrid(y, x)
    mu = 0.1
    f = np.exp(-((x - 0.5 * Lx)**2 + (y - 0.5 * Ly)**2) / mu**2) / mu**2
    g = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    h = g * f
    return h


def wave_poly_trig2d(x, y, Lx, Ly):
    """
    Polynomial times trigonometric wave.
    f_init = 4x^2*y*(1-x)*(1-y)*sin(2pi*x)*cos(2pi*y)
    """
    y, x = np.meshgrid(y, x)
    mu = 0.1
    f = np.exp(-((x - 0.5 * Lx)**2 + (y - 0.5 * Ly)**2) / mu**2) / mu**2
    g = f = 4 * x**2 * y * (1 - x) * (1 - y)
    h = g * f
    return h


def init_vel2d(x, y):
    """Initial wave velocity. Almost always set to 0 for our purposes."""
    return np.zeros((x.shape[0], y.shape[0]))
