# wizard.py
"""
Single function for full starcal process.
Using this will skip some of the more flexible steps.
"""

import numpy as np
import h5py

from asistarcalibration.starcal import StarCal
from asistarcalibration.starfinder import StarFinder

def wizard(img, glat, glon, time, 
           starlist='starlist.txt', 
           outfile='pixelcoords.h5', 
           projalt=110., 
           elevcutoff=15.):

    find = StarFinder(glat, glon, time)
    
    find.find_stars(img)
    
    find.save_starcal_file(starlist)
    
    cal = StarCal(starlist)
    
    cal.calculate_calibration_params(*img.shape)
    
    cal.checkcal(img, glat)
    
    azmt, elev, lat, lon = cal.calculate_position_array(glat, glon, projalt, *img.shape)
    
    mask = elev<np.deg2rad(elevcutoff)
    
    with h5py.File(outfile, 'w') as h5:

        ds = h5.create_dataset('Latitude', data=lat)
        ds.attrs['Description'] = 'Geodetic Latitude'
        ds.attrs['Units'] = 'degrees'
        ds.attrs['Altitude'] = projalt

        ds = h5.create_dataset('Longitude', data=lon)
        ds.attrs['Description'] = 'Geodetic Longitude'
        ds.attrs['Units'] = 'degrees'
        ds.attrs['Altitude'] = projalt

        ds = h5.create_dataset('Azimuth', data=azmt)
        ds.attrs['Description'] = 'Azimuth angle east of North'
        ds.attrs['Units'] = 'radians'

        ds = h5.create_dataset('Elevation', data=elev)
        ds.attrs['Description'] = 'Elevation angle'
        ds.attrs['Units'] = 'radians'

        ds = h5.create_dataset('Mask', data=mask)
        ds.attrs['Description'] = 'Low-elevation mask'
        ds.attrs['Cut-off'] = elevcutoff


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



