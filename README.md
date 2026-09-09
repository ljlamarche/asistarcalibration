# asistarcalibration
This package includes general procedures for geometric or "star" calibration of all-sky imagers. This is a somewhat labor intenseive process that requires manually identifying stars in an image, but the package includes widgets to simplify the process as well as the code to perform the fits to calculate the necessary rotation angle and lens function.  The code is intentionally designed to be generic and takes 2D arrays as image input rather than internally handling specific file formats.

## Installation
This package can be installed from GitHub with pip.  Either clone and install
```
git clone https://github.com/ljlamarche/asistarcalibration.git
cd asistarcalibration
pip install .
```
or install directly from GitHub.
```
pip install git+https://github.com/ljlamarche/asistarcalibration.git
```

## Usage
A full description of the star calibration procedure and explination of both the steps and the background is provided in the [tutorial](https://github.com/ljlamarche/asistarcalibration/blob/main/tutorial/tutorial.ipynb).  The following is provied for quick refrence, but I highly recommend new users refer to the tutorial for detailed instructions.

### Star Finding
Stars are manually selected in an image.  Becaue each camera system is different, the process of reading in an image from a raw data file and manipulating it so the stars are visable is left to the user.  The following example shows how to use the `StarFinder` class after an image (`img`) is prepared.

```
import numpy as np
from asistarcalibration.starfinder import StarFinder

time = np.datetime64('2025-01-03T03:27:00')
glat = 65.5
glon = -147.7

sf = starfinder.StarFinder(glat, glon, time)
sf.find_stars(img)
sf.save_starcal_file('output_starfile.txt')
```
The output file contains a list of identified stars and their pixel locations within the image and will be used in the calibration step.

Suggustions for making stars visable in an image:
- Co-add multiple images taken over a few minutes
- Adjust the contrast in the image
- Histogram equalize the image (`from asistarcalibration.wizard import equalize` provides a simple routine for this)
- Manually select `vmin` and `vmax` for plotting (these can be passed to `find_stars` using the optional `imshow_kw` dictionary input)
- Inverting or "flipping" the image so it matches the orientation in the reference starmaps (this is necessary for some cameras depending on the configuration of lenses)

### Calibration
After a star file has been created, the `StarCal` clsass can be used to calculate the calibration parameters.

```
sc = starcal.StarCal('output_starfile.txt')
sc.calculate_calibration_params(*img.shape)

print(sc.x0, sc.y0, sc.rl, sc.theta, sc.A, sc.B, sc.C, sc.D)
```

The `checkcal` function can be used as a sainity check that the calibration is reasonable.

```
sc.checkcal(img, glat)
```

### Calculate Pixel Positions
The `StarCal` class can also calulate an array of the azimuth and elevation of every pixel in the image, as well as latitude and longitude of each pixel at an assumed altitude.

```
alt = 110.  # km
az, el, lat, lon = sc.calculate_position_array(glat, glon, alt, *img.shape)
```

**Notes:**
- Star identification is based off the [Hipparcos catolog](https://www.cosmos.esa.int/web/hipparcos/catalogues) and work with stars and planets. Stars MUST be identified by their HIP number, but planets can be entered by name.
- Stellarium appends the HIP number of some stars with alphabetical characters, usually indicating a binary star system.  Only enter the numerical digits associated with each star at the command line.
- If the figure that `StarCal` pops up has a star that is wildly out of place, manually edit the starcal file and delete that line.  These can offset the fit and result in bad calibration.  This is often the result of accidentally entering the wrong HIP number.
- You can load the stars from an existing star file using the `load_stars` function after the `StarFinder` class has been initalized.  This is useful if you saved a starcal file midway through working on it, or realize you need to add more stars to improve the fit.

## Funding Acknowlegements
The development of this software has been supported by the following:
- NSF award AGS-1933013
- NSF award AGS-2426523
- NASA award 80NSSC24M0028
