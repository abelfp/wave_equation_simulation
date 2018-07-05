#!/usr/bin/env python3
'''
wave_class.py - Classes for the numerical solution of the wave equation in
1D and 2D in cartesian coordinates.

The classes employ a numerical method that only stores 3 adjacent solutions
in time, making it more efficient in comparison to solving the wave equation
at a given time range.

11Jun18 - Abel Flores Prieto & Sang Hun Chou
'''

import numpy as np
import sys

from . import initial_func as funcs


class Wave1D():
    """Wave Equation in Cartesian Coor. Solver - 1D

    Numerical solution to wave equation with boundary conditions of 0 at both
    end points. This solution uses the method of central differences.
    """

    def __init__(self,
                 u_x0,                       # initial position function
                 ut_x0=funcs.init_vel1d,     # initial velocity function
                 dt=0.001,                   # time step
                 L=10.0,                  # meters, length of string
                 dx=0.005,                   # x step
                 c=1):                       # velocity of wave
        self.x = np.arange(0, L + dx, dx)
        self.alpha2 = (c * dt / dx)**2
        # check if solution will be stable using von Neumann Stability Analysis
        if self.alpha2 > 1:
            print("Your combination of c, dt, and/or dx are not allowed.")
            print("These will make the solution diverge.")
            print("Exiting program...")
            sys.exit()

        self.time = 0
        self.dt = dt

        # zeroth time
        self.u_tim1 = u_x0(self.x, L)
        self.bound_standing(self.u_tim1)

        # first time
        self.u_ti = dt * ut_x0(self.x) + (1 - self.alpha2) * self.u_tim1 + \
            0.5 * self.alpha2 * (np.roll(self.u_tim1, 1) + np.roll(self.u_tim1, -1))
        self.bound_standing(self.u_ti)
        self.time_update()

        # create next grid
        self.u_tip1 = np.zeros(self.u_ti.shape)

    def ux_init(self):
        """
        If called before self.iteration(), then it returns the initial
        condition of u along x. Otherwise, it returns the solution u at
        that given time (along with x which does not change).
        """
        return self.u_tim1, self.x

    def time_update(self):
        """Update the time of the solution"""
        self.time += self.dt

    def bound_standing(self, grid):
        """Boundary conditions for u equal to 0 at all sides."""
        grid[0] = 0
        grid[-1] = 0

    def iteration(self, steps=1):
        """
        Iteration of the solution u. Updates the three arrays u_tim1, u_ti and
        u_tip1 as time progresses by step dt.
        Yields u_ti for time self.time, and self.time.
        """
        while True:
            for _ in range(steps):
                self.u_tip1 = -self.u_tim1 + 2 * (1 - self.alpha2) * self.u_ti + \
                    self.alpha2 * (np.roll(self.u_ti, 1) + np.roll(self.u_ti, -1))
                self.bound_standing(self.u_tip1)
                self.time_update()

                self.u_tim1 = self.u_ti.copy()
                self.u_ti = self.u_tip1.copy()

            yield self.u_ti, self.time


class Wave2D():
    """Wave Equation in Cartesian Coor. Solver - 2D

    Numerical solution to wave equation with boundary conditions of 0 at all
    sides of shape. This solution uses the method of central differences.
    """

    def __init__(self,
                 u_xy0,                       # initial position function
                 ut_xy0=funcs.init_vel2d,  # initial velocity function
                 Lx=1.0,                      # meters, width
                 Ly=1.0,                      # meters, length of sheet
                 dt=0.005,                # time step
                 dxy=0.01,                # x step and y step
                 c=1):                        # velocity of wave
        self.x = np.arange(0, Lx + dxy, dxy)
        self.y = np.arange(0, Ly + dxy, dxy)
        self.alpha2 = (c * dt / dxy)**2
        # check if solution will be stable using von Neumann Stability Analysis
        if self.alpha2 > 0.5:
            print("Your combination of c, dt, and/or dxy are not allowed.")
            print("These will make the solution diverge.")
            print("Exiting program...")
            sys.exit()

        self.time = 0
        self.dt = dt

        # zeroth time
        self.u_tim1 = u_xy0(self.x, self.y, Lx, Ly)
        self.bound_standing(self.u_tim1)

        # first time step
        self.u_ti = dt * ut_xy0(self.x, self.y) + (1 - 2 * self.alpha2) * \
            self.u_tim1 + 0.5 * self.alpha2 * (np.roll(self.u_tim1, 1, axis=0) +
                                               np.roll(self.u_tim1, -1, axis=0) + np.roll(self.u_tim1, 1, axis=1) +
                                               np.roll(self.u_tim1, -1, axis=1))
        self.bound_standing(self.u_ti)
        self.time_update()

        # create next step grid
        self.u_tip1 = np.zeros(self.u_ti.shape)

    def uxy_init(self):
        """
        If called before self.iteration(), then it returns the initial
        condition of u along x and y. Otherwise, it returns the solution u at
        that given time (along with x and y which do not change).
        """
        return self.u_tim1, self.x, self.y

    def time_update(self):
        """Update the time of the solution"""
        self.time += self.dt

    def bound_standing(self, grid):
        """Boundary conditions for u equal to 0 at all sides."""
        grid[0, :] = 0
        grid[:, 0] = 0
        grid[-1, :] = 0
        grid[:, -1] = 0

    def iteration(self, steps=1):
        """
        Iteration of the solution u. Updates the three arrays u_tim1, u_ti and
        u_tip1 as time progresses by step dt.
        Yields u_ti for time self.time, and self.time.
        """
        while True:
            for _ in range(steps):
                self.u_tip1 = -self.u_tim1 + 2 * (1 - 2 * self.alpha2) * self.u_ti + \
                    self.alpha2 * (np.roll(self.u_ti, 1, axis=0) + np.roll(
                        self.u_ti, -1, axis=0) + np.roll(self.u_ti, 1, axis=1) +
                        np.roll(self.u_ti, -1, axis=1))
                self.bound_standing(self.u_tip1)
                self.time_update()

                self.u_tim1 = self.u_ti.copy()
                self.u_ti = self.u_tip1.copy()

            yield self.u_ti, self.time
