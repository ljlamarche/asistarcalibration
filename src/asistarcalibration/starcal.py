#!/usr/bin/env python
"""Find and/or load stars to use for calibration and calculate calibration parameters"""

import datetime as dt
import argparse
import io
import logging
import os
import sys
import requests

import h5py
import numpy as np
import pandas as pd

# Workaround due to bug in matplotlib event handling interface
# https://github.com/matplotlib/matplotlib/issues/30419
import matplotlib as mpl
if mpl.get_backend() == 'macosx':
    mpl.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos
from skyfield.named_stars import named_star_dict

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class StarCal:
    """Star calibration"""

    def __init__(self, glat, glon, time, station=None, instrument=None):

        #self.time = time
        self.site_lat = glat
        self.site_lon = glon
        self.time = time

        self.site_station = str(station)
        self.site_instrument = str(instrument)

        self.starlist = pd.DataFrame(columns=['Name','HIP','az','el','x','y'])


    def load_stars(self, sc_file):

        new_stars = pd.read_table(sc_file, comment='#', sep='\s+')

        self.starlist = pd.concat([self.starlist, new_stars])


    def add_star(self, click):
        """Add user selected star and az,el based on HIP."""
        # This makes use of the Hipparcos Catolog
        # https://rhodesmill.org/skyfield/stars.html

        # Star location in figure from click event
        x = click.xdata
        y = click.ydata
        print(f"Star at {x=:02f}, {y=:02f}")

        # User entered planet or HIP number
        key = input('Planet/HIP: ')

        try:
            # Look up star based on HIP
            if key.isdigit():
                s = Star.from_dataframe(self.hipcat.loc[int(key)])
                name = self.hipcat.loc[int(key),'name']
                hip = int(key)
            # Look up planet
            else:
                try:
                    s = self.planets[key]
                except KeyError:
                    key1 = f'{key} barycenter'
                    s = self.planets[key1]
                name = key
                hip = 0
        except KeyError:
            print(f'Entered {key} is not a recognized planet or in the Hipparcos database!')
            return

        # Calculate az/el
        elev, azmt, _ = self.site_ref.observe(s).apparent().altaz()

        # Append star information
        new_star = {'Name': name,
                    'HIP': hip,
                    'az': azmt.degrees,
                    'el': elev.degrees,
                    'x': x,
                    'y': y}
        self.starlist.loc[len(self.starlist)] = new_star

        # Mark star on plot
        self.ax.scatter(x, y, facecolors='none', edgecolors='r')
        self.fig.canvas.draw()


    def find_stars(self, image):
        """Display image and track manual selection of stars"""

        self.prep_star_lookup(self.time)

        print('Site Information\n'+16*'=')
        print(f'{self.site_station.upper()}    {self.site_instrument}')
        print(f'TIME: {self.time}')
        print(f'GLAT: {self.site_lat}\nGLON: {self.site_lon}')

        # Display image with stars
        self.fig, self.ax = plt.subplots()
        # Set up button press event trigger
        self.fig.canvas.mpl_connect('button_press_event', self.add_star)
        # Display image
        self.ax.imshow(image, cmap='gray')
        
        self.ax.scatter(self.starlist['x'], self.starlist['y'], facecolors='none', edgecolors='r')

        plt.show()


    def prep_star_lookup(self, time):
        """Prepare skyfield for star lookups"""

        # Define site location
        ts = load.timescale()
        t = ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second)
        self.planets = load('de421.bsp')
        earth = self.planets['earth']
        site = earth + wgs84.latlon(self.site_lat, self.site_lon, elevation_m=0)
        self.site_ref = site.at(t)

        # Load HIP catolog
        with load.open(hipparcos.URL) as f:
            df = hipparcos.load_dataframe(f)

        # Add star names to HIP catolog
        star_name_list = ['xxxxx']*len(df)
        df['name'] = star_name_list
        for name, hip in named_star_dict.items():
            df.loc[hip,'name'] = name

        self.hipcat = df

    def check_stars(self):
        """Check stars loaded from input starcal file against HIP catalog for consistency"""
        # Need to look at how this is done to still check input stars

        for name, hip, azel in zip(self.star_name, self.star_hip, self.star_azel):
            print('\nHIP: ', hip)
            print('Name: ', self.hipcat.loc[hip]['name'], name)
            s = Star.from_dataframe(self.hipcat.loc[hip])
            elev, azmt, _ = self.site_ref.observe(s).apparent().altaz()
            print('Azimuth: ', azmt.degrees, azel[0], azmt.degrees-azel[0])
            print('Elevation: ', elev.degrees, azel[1], elev.degrees-azel[1])


    def save_starcal_file(self, output):
        """ Save output starcal file"""

        with open(output, 'w') as f:
            # write header
            f.write(f'# {self.site_station.upper()}    {self.site_instrument}\n')
            f.write(f'# {self.time.isoformat()}\n')
            f.write(f'# GLAT={self.site_lat:12.6f}    GLON={self.site_lon:12.6f}\n')
            f.write(80*'#'+'\n')
            df_string = self.starlist.to_string(header=True, index=False, col_space=[15,8,15,15,10,10])
            f.write(df_string)


    def calculate_calibration_params(self):
        """Load calibration parameters from starcal file"""

        # true x,y positions of stars
        xp = np.cos(self.starlist['el'] * np.pi / 180.0) * np.sin(self.starlist['az'] * np.pi / 180.0)
        yp = np.cos(self.starlist['el'] * np.pi / 180.0) * np.cos(self.starlist['az'] * np.pi / 180.0)

        init_params = self.initial_params(self.starlist['x'], self.starlist['y'], xp, yp)
        params = least_squares(self.residuals, init_params, args=(self.starlist['x'], self.starlist['y'], xp, yp))
        self.x0, self.y0, self.rl, self.theta, self.C, self.D = params.x

        # NOTE: A and B are fully constrained when fitting for rl
        self.A = np.pi / 2.0
        self.B = -(np.pi / 2.0 + self.C + self.D)

        # DEBUG: To confirm star locations match after transformation
        xt, yt = self.transform(self.starlist['x'], self.starlist['y'], self.x0, self.y0, self.rl,
                                  self.theta, self.A, self.B, self.C, self.D)
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        cmap=plt.get_cmap('tab20')
        rp = np.sqrt((self.starlist['x']-self.x0)**2+(self.starlist['y']-self.y0)**2)/self.rl
        for i in range(len(self.starlist)):
            ax1.scatter(xp[i], yp[i], s=5, color=cmap(i%20))    # Projected true star position
            ax1.scatter(xt[i], yt[i], facecolors='none', edgecolors=cmap(i%20)) # Transformed CCD star position
            ax2.scatter(rp[i], self.starlist['el'][i], color=cmap(i%20), label=self.starlist['HIP'][i])
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

#    #def save_calibration_params(self, output, config_file):
#    def save_calibration_params(self, output, config_file):
#        """Save results"""
#
#        params = dict(x0 = str(self.x0),
#                      y0 = str(self.y0),
#                      rl = str(self.rl),
#                      theta = str(self.theta),
#                      a = str(self.A),
#                      b = str(self.B),
#                      c = str(self.C),
#                      d = str(self.D))
#
#        config = configparser.ConfigParser()
#
#        # If a real filename is provide for a config file, read it in
#        if config_file:
#            config.read(config_file)
#
#        config["CALIBRATION_PARAMS"] = dict(
#                x0 = str(self.x0),
#                y0 = str(self.y0),
#                rl = str(self.rl),
#                theta = str(self.theta),
#                a = str(self.A),
#                b = str(self.B),
#                c = str(self.C),
#                d = str(self.D))
#
#
#        with open(output, "w", encoding="utf-8") as cf:
#            config.write(cf)


    def elev2r(self, elev):

        el = np.deg2rad(elev)

        Delta0 = self.C**2 - 3 * self.D * self.B
        Delta1 = 2 * self.C**3 - 9 * self.D * self.C * self.B + 27 * self.D**2 * (self.A-el)
        Gamma = ((Delta1 + np.sqrt(Delta1**2 - 4 * Delta0**3)) / 2)**(1./3.)
        r = -(self.C + Gamma + Delta0/Gamma)/(3 * self.D)

        return r


    def checkcal(self, image):

        # Display image with stars
        fig, ax = plt.subplots()
        # Display image
        ax.imshow(image, cmap='gray')

        # Plot Zenith
        ax.scatter(self.x0, self.y0, s=50, color='red', marker='P', label='zenith')

        # Generate lense function arrays for interpretation
        t = np.linspace(0., 2*np.pi, 100)
        r = np.linspace(0., 1., 100)
        lam = np.rad2deg(self.A + self.B * r + self.C * r**2 + self.D * r**3)

        # Set Up color maps
        cmap = mpl.colormaps['rainbow']
        norm = mpl.colors.Normalize(vmin=0., vmax=90.)
        cmap2 = mpl.colormaps['twilight']
        norm2 = mpl.colors.Normalize(0., 360.)

        # Plot elevation circles
        t = np.linspace(0., 2*np.pi, 100)
        el0 = [0., 15., 30., 45., 60., 75.]
        r0 = self.elev2r(el0)
        for r, el in zip(r0, el0):
            x = r * self.rl * np.cos(t) + self.x0
            y = r * self.rl * np.sin(t) + self.y0
            ax.plot(x, y, color=cmap(el/90.), label=f'el={el}')

        # Plot North Line
        x1 = self.rl * np.sin(np.deg2rad(self.theta)) + self.x0
        y1 = self.rl * np.cos(np.deg2rad(self.theta)) + self.y0
        ax.plot([self.x0, x1], [self.y0, y1], color='k', linestyle=':', label='North')

        # Plot Polaris
        r0 = self.elev2r(self.site_lat)
        x = r0 * self.rl * np.sin(np.deg2rad(self.theta)) + self.x0
        y = r0 * self.rl * np.cos(np.deg2rad(self.theta)) + self.y0
        ax.scatter(x, y, s=50, color='magenta', marker='*', label='Polaris')

        # Add colorbars
        c = ax.scatter(self.starlist['x'], self.starlist['y'], facecolor=cmap2(self.starlist['az']/360.), edgecolor=cmap(self.starlist['el']/90.))
        cax = fig.add_axes([0.8, 0.1, 0.02, 0.8])
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='Elevation - Edge (deg)')
        cax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2), cax=cax, label='Azimuth - Face (deg)')
        # Add legend
        ax.legend()

        plt.show()


def equalize(image, contrast, num_bins=10000):
    """Histogram Equalization to adjust contrast [1%-99%]"""
    # copied function from imageops.py
    # needed to make the image visable - there may be more efficient ways of doing this

    image_array_1d = image.flatten()

    image_histogram, bins = np.histogram(image_array_1d, num_bins)
    image_histogram = image_histogram[1:]
    bins = bins[1:]
    cdf = np.cumsum(image_histogram)

    # spliced to cut off non-image area
    # any way to determine this dynamically?  How periminant is it?
    cdf = cdf[:9996]

    max_cdf = max(cdf)
    max_index = np.argmin(abs(cdf - contrast / 100 * max_cdf))
    min_index = np.argmin(abs(cdf - (100 - contrast) / 100 * max_cdf))
    vmax = float(bins[max_index])
    vmin = float(bins[min_index])
    low_value_indices = image_array_1d < vmin
    image_array_1d[low_value_indices] = vmin
    high_value_indices = image_array_1d > vmax
    image_array_1d[high_value_indices] = vmax

    return image_array_1d.reshape(image.shape)





# ------------------------------------------------------------------------
# Main application
# ------------------------------------------------------------------------


#def parse_args():
#    """Command line parsing"""
#
#    parser = argparse.ArgumentParser(
#        description="Manually identify stars for calibration"
#    )
#
#    parser.add_argument("station", help="Station code")
#    parser.add_argument("instrument", help="redline or greenline")
#    parser.add_argument("-n", "--new", action="store_true", default=False, help="Generate new file from scratch")
#    parser.add_argument("-t", "--time", help="Time for star idenfication")
#
#    parser.add_argument(
#        "-s", "--starcal", metavar="FILE", help="Existing starcal file (for appending stars)"
#    )
#    parser.add_argument(
#        "-o",
#        "--output",
#        default="mango-starcal.txt",
#        help="Output starcal filename (default is mango-starcal.txt)",
#    )
#    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
#
#    return parser.parse_args()
#
#
#def find_starcal(station, instrument):
#    """Find starcal file in package data"""
#
#    # Placeholder for default config file location
#    #   This function can be rewritten later
#    config_dir = os.environ['MANGONETWORK_CONFIGS']
#
#    starcal_file = os.path.join(config_dir, f"starcal-{station}-{instrument}.txt")
#
#    logging.debug("Using package starcal file: %s", starcal_file)
#
#    #return resources.files("mangonetwork.raw.data").joinpath(starcal_file).read_text()
#    return starcal_file
#
#
#def read_header(starcal_file):
#    """Read header from starcal file"""
#
#    with open(starcal_file, 'r') as f:
#        line1 = f.readline()
#        _, station, instrument = line1.split()
#        line2 = f.readline()
#        time = dt.datetime.fromisoformat(line2.split()[1])
#
#    return station, instrument, time
#
#
#def download_image(station, instrument, time):
#    """Download image for star matching"""
#
#    url = f'https://data.mangonetwork.org/data/transport/mango/archive/{station.lower()}/{instrument}/raw/{time:%Y}/{time:%j}/{time:%H}/mango-{station.lower()}-{instrument}-{time:%Y%m%d-%H%M%S}.hdf5'
#    logging.debug("Downloading raw image file: %s", url)
#    r=requests.get(url)
#    open('mango_image.hdf5', 'wb').write(r.content)
#
#    return 'mango_image.hdf5'
#
#
#def main():
#    """Main application"""
#
#    args = parse_args()
#
#    if args.verbose:
#        logging.basicConfig(level=logging.DEBUG)
#    else:
#        logging.basicConfig(level=logging.INFO)
#
#    if args.new:
#        # If new flag set, generate a fresh starcal file
#        logging.debug("Generating new starcal file")
#        station = args.station
#        instrument = args.instrument
#        time = dt.datetime.fromisoformat(args.time)
#        starcal_file = None
#
#    elif args.starcal:
#        # If starcal file provided, read in header
#        if not os.path.exists(args.starcal):
#            logging.error("Starcal file not found")
#            sys.exit(1)
#        logging.debug("Using provided starcal file: %s", args.starcal)
#        starcal_file = args.starcal
#        station, instrument, time = read_header(starcal_file)
#
#    else:
#        # If no starcal file provided, find the default and read in header
#        starcal_file = find_starcal(args.station, args.instrument)
#        if not os.path.exists(starcal_file):
#            logging.error("No default starcal file found for %s %s!", args.station, args.instrument)
#            sys.exit(1)
#        logging.debug("Using defalt starcal file: %s", starcal_file)
#        station, instrument, time = read_header(starcal_file)
#
#    # Download image
#    image_filename = download_image(station, instrument, time)
#
#    # Run star calibration
#    StarCal(image_filename, args.output, sc_file=starcal_file)
#
#    sys.exit(0)
#
#
#if __name__ == "__main__":
#    main()
