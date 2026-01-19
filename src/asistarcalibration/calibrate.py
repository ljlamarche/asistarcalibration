#!/usr/bin/env python
"""Calibration"""

##########################################################################
#
#   Calibration
#
#   2022-xx-xx  Leslie Lamarche and Asti Bhatt
#               Initial implementation
#
#   2023-03-08  Todd Valentic
#               Make PEP8 compliant
#
##########################################################################

import argparse
import configparser
import io
import logging
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

#if sys.version_info < (3, 9):
#    import importlib_resources as resources
#else:
#    from importlib import resources


class Calibrate:
    """Camera calibration"""

    #def __init__(self, config_file, starcal_file, output):
    def __init__(self, starcal_file, output, config=None):
        self.find_calibration_params(starcal_file)
        self.save_calibration_params(output, config)
        #self.save_calibration_params(output)

    def find_calibration_params(self, starcal_file):
        """Load calibration parameters from starcal file"""

        # read in data from starcal file
        star_num, star_az, star_el, x, y = np.loadtxt(starcal_file, unpack=True, usecols=(1,2,3,4,5))

        # true x,y positions of stars
        xp = np.cos(star_el * np.pi / 180.0) * np.sin(star_az * np.pi / 180.0)
        yp = np.cos(star_el * np.pi / 180.0) * np.cos(star_az * np.pi / 180.0)

        init_params = self.initial_params(x, y, xp, yp)
        params = least_squares(self.residuals, init_params, args=(x, y, xp, yp))
        self.x0, self.y0, self.rl, self.theta, self.C, self.D = params.x

        # NOTE: A and B are fully constrained when fitting for rl
        self.A = np.pi / 2.0
        self.B = -(np.pi / 2.0 + self.C + self.D)

        # DEBUG: To confirm star locations match after transformation
        xt, yt = self.transform(x, y, self.x0, self.y0, self.rl,
                                  self.theta, self.A, self.B, self.C, self.D)
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        cmap=plt.get_cmap('tab20')
        rp = np.sqrt((x-self.x0)**2+(y-self.y0)**2)/self.rl
        for i in range(len(star_num)):
            ax1.scatter(xp[i], yp[i], s=5, color=cmap(i%20))    # Projected true star position
            ax1.scatter(xt[i], yt[i], facecolors='none', edgecolors=cmap(i%20)) # Transformed CCD star position
            ax2.scatter(rp[i], star_el[i], color=cmap(i%20), label=int(star_num[i]))
        ax1.set_xlabel(r'$X/R_L$')
        ax1.set_ylabel(r'$Y/R_L$')
        ax1.set_title('Alignment of Transformed Star Position\nwith True Projected Position')
        ax1.grid()

        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Projected True Position', markerfacecolor='k', markersize=5),
                           Line2D([0], [0], marker='o', color='w', label='Transformed CCD Position', markeredgecolor='k', markersize=5)]
        ax1.legend(handles=legend_elements, loc='upper left')
        
        ax2.set_xlabel('R')
        ax2.set_ylabel('Elevation (deg)')
        ax2.set_title('Lens Function')
        ax2.set_xlim([0., 1.])
        ax2.set_ylim([0., 90.])
        ax2.grid()
        theta = np.linspace(0., 2*np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), color='dimgrey')
        r = np.arange(0., 1., 0.01)
        t = self.A + self.B*r + self.C*r**2 + self.D*r**3
        ax2.plot(r, np.rad2deg(t), color='dimgrey')
        lf_str = (f'A={np.rad2deg(self.A):.2f}\n'
                  f'B={np.rad2deg(self.B):.2f}\n'
                  f'C={np.rad2deg(self.C):.2f}\n'
                  f'D={np.rad2deg(self.D):.2f}')
        ax2.text(0.98, 0.98, lf_str, ha='right', va='top', transform=ax2.transAxes)
        ax2.legend(loc='center left', bbox_to_anchor=(1.01,0.5), fontsize='x-small')
        plt.show()

    # pylint: disable=too-many-arguments, too-many-locals

    def transform(self, x, y, x0, y0, rl, theta, A, B, C, D):
        """Transformation"""

        x1 = (x - x0) / rl
        y1 = (y - y0) / rl

        t = theta * np.pi / 180.0
        x2 = np.cos(t) * x1 - np.sin(t) * y1
        y2 = np.sin(t) * x1 + np.cos(t) * y1

        r = np.sqrt(x2**2 + y2**2)
        lam = A + B * r + C * r**2 + D * r**3
        d = np.cos(lam)

        x3 = d * x2 / r
        y3 = d * y2 / r

        return x3, y3

    def residuals(self, params, x, y, xp, yp):
        """Residuals"""

        x0, y0, rl, theta, C, D = params
        A = np.pi / 2.0
        B = -(np.pi / 2.0 + C + D)
        xt, yt = self.transform(x, y, x0, y0, rl, theta, A, B, C, D)
        res = np.sqrt((xp - xt) ** 2 + (yp - yt) ** 2)
        return res

    def initial_params(self, x, y, xp, yp):
        """Initial parameters"""

        # Use center of image and half of y distance for x0, y0, and rl
        x0, y0, rl = [347.5, 259.5, 259.5]

        # appriximate lense function with line
        A, B, C, D = [np.pi / 2, -np.pi / 2, 0.0, 0.0]

        # calculate transformation with initial tranlation and lens function params but no rotation
        xu, yu = self.transform(x, y, x0, y0, rl, 0.0, A, B, C, D)

        # Find rotation matrix such that the vectors to the star locations roughly match
        Pu = np.array([xu, yu, np.zeros(len(x))]).T
        Pp = np.array([xp, yp, np.zeros(len(x))]).T
        R, _ = Rotation.align_vectors(Pp, Pu)

        # Find euler angles of rotation matrix and select "z" rotation as an approximate theta
        theta = R.as_euler("xyz", degrees=True)[2]

        return [x0, y0, rl, theta, C, D]

    #def save_calibration_params(self, output, config_file):
    def save_calibration_params(self, output, config_file):
        """Save results"""

        params = dict(x0 = str(self.x0),
                      y0 = str(self.y0),
                      rl = str(self.rl),
                      theta = str(self.theta),
                      a = str(self.A),
                      b = str(self.B),
                      c = str(self.C),
                      d = str(self.D))

        config = configparser.ConfigParser()

        # If a real filename is provide for a config file, read it in
        if config_file:
            config.read(config_file)

        config["CALIBRATION_PARAMS"] = dict(
                x0 = str(self.x0),
                y0 = str(self.y0),
                rl = str(self.rl),
                theta = str(self.theta),
                a = str(self.A),
                b = str(self.B),
                c = str(self.C),
                d = str(self.D))


        with open(output, "w", encoding="utf-8") as cf:
            config.write(cf)


# -------------------------------------------------------------------------
# Main application
# -------------------------------------------------------------------------


def parse_args():
    """Command line options"""

    parser = argparse.ArgumentParser(description="Calculate camera calibration")

    parser.add_argument("station", help="Station code")
    parser.add_argument("instrument", help="redline or greenline")

    parser.add_argument(
        "-c", "--config", metavar="FILE", help="Existing configuration file"
    )
    parser.add_argument(
        "-s", "--starcal", metavar="FILE", help="Starcal file"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mango-config.ini",
        help="Output configuration filename (default is mango-config.ini)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def find_config(station, instrument):
    """Find configuration file from pacakge data"""

    # Placeholder for default config file location
    #   This function can be rewritten later
    config_dir = os.environ['MANGONETWORK_CONFIGS']

    config_file = os.path.join(config_dir, f"{station}-{instrument}.ini")

    logging.debug("Using package configuration file: %s", config_file)

    #return resources.files('mangonetwork.raw.data').joinpath(config_file).read_text()
    return config_file


def find_starcal(station, instrument):
    """Find starcal file in package data"""

    # Placeholder for default config file location
    #   This function can be rewritten later
    config_dir = os.environ['MANGONETWORK_CONFIGS']

    starcal_file = os.path.join(config_dir, f"starcal-{station}-{instrument}.txt")

    logging.debug("Using package starcal file: %s", starcal_file)

    #return resources.files("mangonetwork.raw.data").joinpath(starcal_file).read_text()
    return starcal_file


def main():
    """Main application"""

    args = parse_args()

    fmt = "[%(asctime)s] %(levelname)s %(message)s"

    if args.verbose:
        logging.basicConfig(format=fmt, level=logging.DEBUG)
    else:
        logging.basicConfig(format=fmt, level=logging.INFO)

    if args.config:
        logging.debug("Alternate configuration file: %s", args.config)
        # If configuration file provided, check that it exists
        if not os.path.exists(args.config):
            logging.error("Configurationl file not found")
            sys.exit(1)
        logging.debug("Using provided configuration file: %s", args.config)
        config_file = args.config
    else:
        config_file = None

    # Determine starcal filename
    if args.starcal:
        # If starcal file specified, check that it exists
        logging.debug("Alternate starcal file: %s", args.starcal)
        if not os.path.exists(args.starcal):
            logging.error("Starcal file not found")
            sys.exit(1)
        logging.debug("Using provided starcal file: %s", args.starcal)
        starcal_file = args.starcal
    else:
        # If no starcal file provided, find the default
        starcal_file = find_starcal(args.station, args.instrument)
        if not os.path.exists(starcal_file):
            logging.error("No default starcal file found for %s %s!", args.station, args.instrument)
            sys.exit(1)
        logging.debug("Using defalt starcal file: %s", starcal_file)


    Calibrate(starcal_file, args.output, config=config_file)

    sys.exit(0)


if __name__ == "__main__":
    main()
