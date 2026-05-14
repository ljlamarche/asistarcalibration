# starfind_example.py
# minimal example to run StarFinder

import numpy as np
from PIL import Image
from skimage import exposure
from asistarcalibration import starcal, starfinder

filelist = ['PFRR_20250103_032703_0558.png',
            'PFRR_20250103_032711_0558.png',
            'PFRR_20250103_032719_0558.png',
            'PFRR_20250103_032728_0558.png',
            'PFRR_20250103_032737_0558.png',
            'PFRR_20250103_032745_0558.png',
            'PFRR_20250103_032753_0558.png',
            'PFRR_20250103_032802_0558.png',
            'PFRR_20250103_032810_0558.png',
            'PFRR_20250103_032818_0558.png',
            'PFRR_20250103_032826_0558.png',
            'PFRR_20250103_032834_0558.png',
            'PFRR_20250103_032842_0558.png',
            'PFRR_20250103_032851_0558.png',
           ]

imglist = list()
for file in filelist:
    im = Image.open(file)
    imglist.append(im)
img = np.mean(imglist, axis=0)
img_eq = exposure.equalize_hist(img)

time = np.datetime64('2025-01-03T03:27:00')
glat = 65.5
glon = -147.7

sf = starfinder.StarFinder(glat, glon, time)
sf.find_stars(img_eq)
sf.save_starcal_file('test_out.txt')