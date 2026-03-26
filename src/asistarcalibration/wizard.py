# wizard.py
"""
Single function for full starcal process.
Using this will skip some of the more flexible steps.
"""

import numpy as np
import h5py
from apexpy import Apex

from asistarcalibration.starcal import StarCal
from asistarcalibration.starfinder import StarFinder

def wizard(img, site_lat, site_lon, time, 
           starlist='starlist.txt', 
           outfile='pixelcoords.h5', 
           projalt=110., 
           elevcutoff=15.,
           footalt=110.,
           starfind = True,
           plot_kw=dict()):

    # Skip starfinding step
    if starfind:
        find = StarFinder(site_lat, site_lon, time)
        
        find.find_stars(img, plot_kw)
        
        find.save_starcal_file(starlist)
    
    cal = StarCal(starlist)
    
    cal.calculate_calibration_params(*img.shape)
    
    cal.checkcal(img, site_lat)
    
    azmt, elev, glat, glon = cal.calculate_position_array(site_lat, site_lon, projalt, *img.shape)
    
    mask = elev<np.deg2rad(elevcutoff)

    ###########################################################################
    # Magnetic mapping/footpointing
    # This may be a specialized thing that should be handled elsewhere
    ###########################################################################

    A = Apex(time.item())
    mlat, mlon = A.geo2apex(glat, glon, projalt)

    if projalt >= footalt:
        # No problem if emission altitude greater than footpoint altitude
        flat, flon, _ = A.apex2geo(mlat, mlon, footalt)
    else:
        # Find the minimum rectangle for the unmasked region
        # This is a work around to deal with the fact that the corners of the image
        #  are so far from the center of the FoV they get mapped to a completetly 
        #  different part of the Earth and are sometimes at low enough magnetic 
        #  latitudes that the field lines do not cross the 110 km footpointing 
        #  altitude.
        nomask = np.argwhere(~mask)
        imin = np.min(nomask[:,0])
        imax = np.max(nomask[:,0])
        jmin = np.min(nomask[:,1])
        jmax = np.max(nomask[:,1])
        # This could also be handled by masking the mlat/mlon arrays outside the FoV
        flat, flon, _ = A.apex2geo(mlat[imin:imax,jmin:jmax], mlon[imin:imax,jmin:jmax], footalt)

#    flat, flon = A.map_to_height(glat, glon, projalt, footalt)

    ###########################################################################
    
    with h5py.File(outfile, 'w') as h5:

        ds = h5.create_dataset('Latitude', data=glat)
        ds.attrs['Description'] = 'Geodetic Latitude'
        ds.attrs['Units'] = 'degrees'
        ds.attrs['Altitude'] = projalt

        ds = h5.create_dataset('Longitude', data=glon)
        ds.attrs['Description'] = 'Geodetic Longitude'
        ds.attrs['Units'] = 'degrees'
        ds.attrs['Altitude'] = projalt

        ds = h5.create_dataset('MagneticLatitude', data=mlat)
        ds.attrs['Description'] = 'Apex Magnetic Latitude'
        ds.attrs['Units'] = 'degrees'

        ds = h5.create_dataset('MagneticLongitude', data=mlon)
        ds.attrs['Description'] = 'Apex Magnetic Longitude'
        ds.attrs['Units'] = 'degrees'

        ds = h5.create_dataset('FootpointLatitude', data=flat)
        ds.attrs['Description'] = 'Magnetic Footpoint Latitude'
        ds.attrs['Units'] = 'degrees'
        ds.attrs['Altitude'] = footalt
        if projalt < footalt:
            ds.attrs['Irange'] = [imin, imax]
            ds.attrs['Jrange'] = [jmin, jmax]
            ds.attrs['TrimDescription'] = 'The corners of the image (outside the FoV) cannot be mapped to the footpoint, so this array has been trimed to the index range given in Irange, Jrange'

        ds = h5.create_dataset('FootpointLongitude', data=flon)
        ds.attrs['Description'] = 'Magnetic Footpoint Longitude'
        ds.attrs['Units'] = 'degrees'
        ds.attrs['Altitude'] = footalt
        if projalt < footalt:
            ds.attrs['Irange'] = [imin, imax]
            ds.attrs['Jrange'] = [jmin, jmax]
            ds.attrs['TrimDescription'] = 'The corners of the image (outside the FoV) cannot be mapped to the footpoint, so this array has been trimed to the index range given in Irange, Jrange'

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



