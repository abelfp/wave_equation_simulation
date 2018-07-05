#!/usr/bin/env python3
'''
wave_project.py - Main script for final project for Phys 129L.

11Jun18 - Abel Flores Prieto & Sang Hun Chou
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from inspect import getmembers, isfunction
from wave_pack import wave_class as waves
from wave_pack import initial_func as funcs


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


def user_1d():
    """
    Asks user for the type of pulse they would like to see.
    """
    # The following two lines create a dictionary of the libraries that can be
    # used for the 1D simulation
    funcs_1d = [i for i in func_list if i[0][:5] == 'pulse']
    func_dict = {n: elem for n, elem in enumerate(funcs_1d)}

    print("So you chose to see one of our 1D demonstrations! Great!")
    print("What initial pulse would you like to see in the animation:")
    print("Enter the number corresponding to that pulse.\n")

    # display possible pulses from dictionary
    for k, v in func_dict.items():
        print("{}: {}".format(k, v[0]))

    # check the user inputs a correct option
    while True:
        try:
            pulse = int(input("\n> "))
            assert pulse in list(func_dict.keys())
            break
        except:
            print("Please enter a valid integer.")

    # show animation calling the function wave1d_ani
    wave1d_ani(func_dict[pulse])


def wave1d_ani(function):
    """
    Animation of 1D Wave Equation for a string with both end points
    bounded to 0 at all times.
    """
    # create the function to animate and update the time in plot
    def animate(data):
        u, t = data
        line.set_ydata(u)  # update the data
        text_time.set_text("time = {:.2f}".format(t))  # update the time
        return line, text_time

    func_name, f_init = function

    # initializing wave class in 1-dimension with initial positional func
    string = waves.Wave1D(f_init, L=8)
    u, x = string.ux_init()  # initial state of wave

    # create figure and put labels
    fig, ax = plt.subplots()
    line, = ax.plot(x, u)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t)$")
    ax.set_title("Traveling Wave on String")
    ax.plot(x[0], 0, 'o', x[-1], 0, 'o', ms=8, color='C0')
    text_time = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.set_ylim(-1.5, 1.5)

    # create animation, string.iteration(50) yields u after 50 time steps
    ani = animation.FuncAnimation(fig, animate, string.iteration(10),
                                  interval=1, blit=True, save_count=1000)

    # ani.save("{}.mp4".format(func_name), fps=60)
    # we use try/except statements because every time we close the animation
    # we get an error which we can clear afterwards.
    try:
        print("To exit animation simply click the 'x' in the corner.")
        plt.show()
    except:
        print("Exiting animation...")
        cls()  # clears screen  # clears the screen in terminal


def user_2d():
    """
    Asks user for the type of 2D wave they would like to see.
    """
    # we create a dictionary of functions that work with the 2D simulations
    funcs_2d = [i for i in func_list if i[0][0:4] == 'wave']
    func_dict = {n: elem for n, elem in enumerate(funcs_2d)}

    print("Moving into the interesting stuff, eh? Awesome!")
    print("What initial wave would you like to see in the animation:")
    print("Enter the number corresponding to that wave.\n")

    for k, v in func_dict.items():
        print("{}: {}".format(k, v[0]))

    # we make sure the user inputs a correct option
    while True:
        try:
            option = int(input("\n> "))
            assert option in list(func_dict.keys())
            break
        except:
            print("Please enter a valid integer.")

    # animate giving the value to the option the user chose
    wave2d_ani(func_dict[option])


def wave2d_ani(function):
    """
    Animation of 2D Wave Equation with all sides bounded to 0 at all times.
    """
    # update function and time for most waves in 2D
    def update(data):
        u, t = data
        ax.clear()
        ax.set_axis_off()
        ax.set_zlim3d(-up, up)
        ax.set_title("time = {:.2f}".format(t))
        surf = ax.plot_surface(x, y, u, cmap=cm.spectral, vmin=-up, vmax=up)
        return surf,

    # separate function for the gaussian since it varies so much from the start
    # to the rest of the animation.
    def update_gauss(data):
        u, t = data
        ax.clear()
        ax.set_axis_off()
        ax.set_zlim3d(-up, up)
        ax.set_title("time = {:.2f}".format(t))
        surf = ax.plot_surface(x, y, u, color='grey', vmin=-up, vmax=up)
        return surf,

    # assigning name and initial function from tuple 'function'
    func_name, f_init = function

    # initializing 2D wave with initial positional function f_init
    wave = waves.Wave2D(f_init, c=1)
    u, x, y = wave.uxy_init()  # initial state of wave
    y, x = np.meshgrid(y, x)  # y first to get the right dimensions to u

    up = u.max()  # how big the wave is at the beginning

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # setting the axis off speeds up the animation
    ax.set_axis_off()
    ax.set_zlim3d(-up, up)
    ax.set_title("time = 0.0")

    # for gaussian we don't use a cmap, but simply a color grey since a cmap
    # would not show anything special
    if func_name == 'wave_gaussian2d':
        surf = ax.plot_surface(x, y, u, color='grey')
        ani = animation.FuncAnimation(fig, update_gauss, wave.iteration(1),
                                      interval=100, save_count=400)
    else:
        surf = ax.plot_surface(x, y, u, cmap=cm.spectral, vmin=-up, vmax=up)
        bar = plt.colorbar(surf, shrink=0.5, aspect=5)
        bar.set_label(r"u(x,y,t)")
        ani = animation.FuncAnimation(fig, update, wave.iteration(1),
                                      interval=100, save_count=400)

    # ani.save("{}.mp4".format(func_name), fps=30)
    # again with the try/except statements to catch the error and clear
    try:
        print("The initial wave was:")
        print(f_init.__doc__.replace("\t", ""))
        print("To exit animation simply click the 'x' in the corner.")
        plt.show()
    except:
        print("Exiting animation...")
        cls()  # clears screen


def start_message():
    """Print start message"""
    cls()  # clears screen
    print("""
Hello and welcome to our project. In this project we will show you some cool
waves in 1 and 2 dimensions.

These are solutions to the wave equation using the method of central
differences.

The animations in this demonstration have been formulated specifically for
a Raspberry Pi because of its specs.

Created by: Abel Flores Prieto and Sang Hun Chou.""")

    input("\nPress <Enter> to continue...")


def main():
    """Start of program."""
    cls()  # clears screen
    while True:
        print("Do you want to explore 1 or 2 dimensions?")
        print("(Enter 'q' to exit the program.)")
        try:
            answer = input("> ")
            if answer.lower()[0] == 'o' or answer[0] == '1':
                user_1d()
            elif answer.lower()[0] == 't' or answer[0] == '2':
                user_2d()
            elif answer.lower()[0] == 'q':  # exit
                print("Thank you for looking through our demonstrations!")
                print("Bye!")
                return
            else:
                print("Sorry, '{}' is not an option.".format(answer))
        except:
            print("Please specify if you want '1' or '2' dimensions.")


if __name__ == "__main__":

    # list of all functions in initial_func module. This is a global variable.
    func_list = [o for o in getmembers(funcs) if isfunction(o[1])]

    # start the interaction through the terminal
    start_message()
    main()
