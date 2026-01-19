#!/usr/bin/env python
"""Plot Star Calibration files"""

##########################################################################
#
#   Plot starcal file
#
#   2022-xx-xx  Leslie Lamarche and Asti Bhatt
#               Initial implementation
#
#   2023-03-08  Todd Valentic
#               PEP8 compliance
#
##########################################################################

import datetime as dt
import argparse
import io
import logging
import os
import sys
import requests

import h5py
import numpy as np

# Workaround due to bug in matplotlib event handling interface
# https://github.com/matplotlib/matplotlib/issues/30419
import matplotlib
if matplotlib.get_backend() == 'macosx':
    matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos
from skyfield.named_stars import named_star_dict


#if sys.version_info < (3, 9):
#    import importlib_resources as resources
#else:
#    from importlib import resources


class StarCal:
    """Star calibration"""

    # load existing stars?
    # Find stars
    # Perform Fit
    # Check calibration
    # Calculate output arrays
    # save output arrays to file

    def __init__(self, image, time, glat, glon, output_file, sc_file=None):

        self.time = time
        self.site_lat = glat
        self.site_lon = glon

        self.site_station = 'PKR'
        self.site_instrument = 'xxx'

        #image = self.load_image(image_file)
        self.find_stars(image, sc_file)
        self.save_starcal_file(output_file)


    def add_star(self, click):
        """Add user selected star and az,el based on HIP."""
        # This makes use of the Hipparcos Catolog
        # https://rhodesmill.org/skyfield/stars.html

        # Star location in figure from click event
        x = click.xdata
        y = click.ydata
        print(f"Star at {x=:02f}, {y=:02f}")

        # User entered HIP number
        hip = input('HIP #: ')
        hip = int(hip)
        print(hip)

        # Look up star based on HIP and calculate az/el
        try:
            s = Star.from_dataframe(self.hipcat.loc[hip])
        except KeyError:
            print(f'Entered Hipparcos designation {hip} is not in database!')
            return

        elev, azmt, _ = self.site_ref.observe(s).apparent().altaz()

        # Append star information
        self.star_azel.append([azmt.degrees, elev.degrees])
        self.star_pos.append([x, y])
        self.star_hip.append(hip)
        self.star_name.append(self.hipcat.loc[hip,'name'])

        # Mark star on plot
        self.ax.scatter(x, y, facecolors='none', edgecolors='r')
        self.fig.canvas.draw()

#    def load_image(self, raw_file):
#        """Load image and metadata from raw file"""
#
#        image = h5py.File(raw_file, 'r')['image']
#        cooked_image = self.prep_image(image)
#
#        self.time = dt.datetime.utcfromtimestamp(image.attrs['start_time'])
#        self.site_lat = image.attrs['latitude']
#        self.site_lon = image.attrs['longitude']
#        self.site_station = image.attrs['station']
#        self.site_instrument = image.attrs['instrument']
#
#        return cooked_image

    def prep_image(self, image, contrast=99.0, rotation_angle=0.):
        """Prepare image to display"""

        cooked_image = np.array(image)
        #cooked_image = equalize(cooked_image, contrast)

        return cooked_image

    def find_stars(self, image, sc_file=None):
        """Display image and track manual selection of stars"""

        self.prep_star_lookup()

        print('Site Information\n'+16*'=')
        #print(f'{self.site_station.upper()}    {self.site_instrument}')
        print(f'TIME: {self.time}')
        print(f'GLAT: {self.site_lat}\nGLON: {self.site_lon}')

        # Initialize arrays
        self.star_name = list()
        self.star_hip = list()
        self.star_azel = list()
        self.star_pos = list()
        if sc_file:
            with open(sc_file) as f:
                for line in f:
                    # Skip lines that are comments or blank
                    if (not line or line[0]=='#' or line.isspace()):
                        continue
                    name, hip, az, el, x, y = line.split()
                    self.star_name.append(name)
                    self.star_hip.append(int(hip))
                    self.star_azel.append([float(az), float(el)])
                    self.star_pos.append([float(x), float(y)])
            # Check stars included in the provided starcal file against hip catolog
            self.check_stars()

        # Display image with stars
        self.fig, self.ax = plt.subplots()
        # Set up button press event trigger
        self.fig.canvas.mpl_connect('button_press_event', self.add_star)
        # Display image
        self.ax.imshow(image, cmap='gray')
        
        for x, y in self.star_pos:
            self.ax.scatter(x, y, facecolors='none', edgecolors='r')

        plt.show()


    def prep_star_lookup(self):
        """Prepare skyfield for star lookups"""

        # Define site location
        ts = load.timescale()
        t = ts.utc(self.time.year,self.time.month,self.time.day,self.time.hour,self.time.minute,self.time.second)
        planets = load('de421.bsp')
        earth = planets['earth']
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

        #filtered_df = df[df['name'] != 'xxxxx']
        #print(filtered_df)
        self.hipcat = df

    def check_stars(self):
        """Check stars loaded from input starcal file against HIP catalog for consistency"""

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
            f.write(80*'#'+'\n\n')
            f.write(f'# {"Name":<20}{"HIP":>8}{"Azimuth":>15}{"Elevation":>15}{"X":>10}{"Y":>10}\n')

            # add new stars
            for name, hip, azel, pos in zip(self.star_name, self.star_hip, self.star_azel, self.star_pos):
                f.write(f'{name:20s}{hip:10d}{azel[0]:15.4f}{azel[1]:15.4f}{pos[0]:10.2f}{pos[1]:10.2f}\n')



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


def parse_args():
    """Command line parsing"""

    parser = argparse.ArgumentParser(
        description="Manually identify stars for calibration"
    )

    parser.add_argument("station", help="Station code")
    parser.add_argument("instrument", help="redline or greenline")
    parser.add_argument("-n", "--new", action="store_true", default=False, help="Generate new file from scratch")
    parser.add_argument("-t", "--time", help="Time for star idenfication")

    parser.add_argument(
        "-s", "--starcal", metavar="FILE", help="Existing starcal file (for appending stars)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mango-starcal.txt",
        help="Output starcal filename (default is mango-starcal.txt)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def find_starcal(station, instrument):
    """Find starcal file in package data"""

    # Placeholder for default config file location
    #   This function can be rewritten later
    config_dir = os.environ['MANGONETWORK_CONFIGS']

    starcal_file = os.path.join(config_dir, f"starcal-{station}-{instrument}.txt")

    logging.debug("Using package starcal file: %s", starcal_file)

    #return resources.files("mangonetwork.raw.data").joinpath(starcal_file).read_text()
    return starcal_file


def read_header(starcal_file):
    """Read header from starcal file"""

    with open(starcal_file, 'r') as f:
        line1 = f.readline()
        _, station, instrument = line1.split()
        line2 = f.readline()
        time = dt.datetime.fromisoformat(line2.split()[1])

    return station, instrument, time


def download_image(station, instrument, time):
    """Download image for star matching"""

    url = f'https://data.mangonetwork.org/data/transport/mango/archive/{station.lower()}/{instrument}/raw/{time:%Y}/{time:%j}/{time:%H}/mango-{station.lower()}-{instrument}-{time:%Y%m%d-%H%M%S}.hdf5'
    logging.debug("Downloading raw image file: %s", url)
    r=requests.get(url)
    open('mango_image.hdf5', 'wb').write(r.content)

    return 'mango_image.hdf5'


def main():
    """Main application"""

    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.new:
        # If new flag set, generate a fresh starcal file
        logging.debug("Generating new starcal file")
        station = args.station
        instrument = args.instrument
        time = dt.datetime.fromisoformat(args.time)
        starcal_file = None

    elif args.starcal:
        # If starcal file provided, read in header
        if not os.path.exists(args.starcal):
            logging.error("Starcal file not found")
            sys.exit(1)
        logging.debug("Using provided starcal file: %s", args.starcal)
        starcal_file = args.starcal
        station, instrument, time = read_header(starcal_file)

    else:
        # If no starcal file provided, find the default and read in header
        starcal_file = find_starcal(args.station, args.instrument)
        if not os.path.exists(starcal_file):
            logging.error("No default starcal file found for %s %s!", args.station, args.instrument)
            sys.exit(1)
        logging.debug("Using defalt starcal file: %s", starcal_file)
        station, instrument, time = read_header(starcal_file)

    # Download image
    image_filename = download_image(station, instrument, time)

    # Run star calibration
    StarCal(image_filename, args.output, sc_file=starcal_file)

    sys.exit(0)


if __name__ == "__main__":
    main()
