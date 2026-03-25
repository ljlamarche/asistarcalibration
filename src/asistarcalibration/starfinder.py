# starfinder.py
"""
GUI to locate and enter stars by clicking in matplotlib image frame.
"""

import datetime as dt
import numpy as np
import pandas as pd

# Workaround due to bug in matplotlib event handling interface
# https://github.com/matplotlib/matplotlib/issues/30419
import matplotlib as mpl
if mpl.get_backend() == 'macosx':
    mpl.use('tkagg')
import matplotlib.pyplot as plt

from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos
from skyfield.named_stars import named_star_dict



class StarFinder:
    """Manual star locating"""

    def __init__(self, glat, glon, time, station=None, instrument=None):

        self.site_lat = glat
        self.site_lon = glon

        # Make sure timw is timezone aware datetime object
        time = time.item()
        if not time.tzinfo:
            self.time = time.replace(tzinfo=dt.timezone.utc)

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

        # For single clicks, do nothing
        if not click.dblclick:
            return

        # Star location in figure from click event
        x = click.xdata
        y = click.ydata
        print(f"Star at {x=:02f}, {y=:02f}")
        new_pnt = self.ax.scatter(x, y, facecolors='blue', edgecolors='none')
        self.fig.canvas.draw()

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
            new_pnt.remove()
            self.fig.canvas.draw()
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
        new_pnt.set(facecolors='none', edgecolors='r')
        self.fig.canvas.draw()


    def find_stars(self, image, imshow_kw=dict()):
        """Display image and track manual selection of stars"""

        self.prep_star_lookup()

        print('Site Information\n'+16*'=')
        print(f'{self.site_station.upper()}    {self.site_instrument}')
        print(f'TIME: {self.time}')
        print(f'GLAT: {self.site_lat}\nGLON: {self.site_lon}')

        # Display image with stars
        self.fig, self.ax = plt.subplots()
        # Set up button press event trigger
        self.fig.canvas.mpl_connect('button_press_event', self.add_star)
        # Display image
        self.ax.imshow(image, cmap='gray', **imshow_kw)
        
        self.ax.scatter(self.starlist['x'], self.starlist['y'], facecolors='none', edgecolors='r')

        plt.show()


    def prep_star_lookup(self):
        """Prepare skyfield for star lookups"""

        # Convert input time to skyfiled time object
        ts = load.timescale()
        t = ts.from_datetime(self.time)

        # Define site location
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


