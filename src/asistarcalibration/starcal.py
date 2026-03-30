#!/usr/bin/env python
"""Find and/or load stars to use for calibration and calculate calibration parameters"""

import numpy as np
import pandas as pd
import pymap3d as pm

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.optimize import least_squares


class StarCal:
    """Star calibration"""

    def __init__(self, starfile):

        self.starlist = pd.DataFrame(columns=['Name','HIP','az','el','x','y'])

        self.load_stars(starfile)
        

    def load_stars(self, sc_file):

        new_stars = pd.read_table(sc_file, comment='#', sep='\s+')

        self.starlist = pd.concat([self.starlist, new_stars])


    def calculate_calibration_params(self, imax, jmax, plot=True):
        """Load calibration parameters from starcal file"""

        # true az/el of stars
        az0 = self.starlist['az'] * np.pi / 180.
        el0 = self.starlist['el'] * np.pi / 180.

        init_params = self.initial_params(self.starlist['x'], self.starlist['y'], az0, el0, imax, jmax)

        params = least_squares(self.residuals, init_params, args=(self.starlist['x'], self.starlist['y'], az0, el0))
        self.x0, self.y0, self.rl, self.theta, self.C, self.D = params.x

        # NOTE: A and B are fully constrained when fitting for rl
        self.A = np.pi / 2.0
        self.B = -(np.pi / 2.0 + self.C + self.D)


        print(f'X0={self.x0}')
        print(f'Y0={self.y0}')
        print(f'RL={self.rl}')
        print(f'THETA={self.theta}')
        print(f'A={np.rad2deg(self.A)}')
        print(f'B={np.rad2deg(self.B)}')
        print(f'C={np.rad2deg(self.C)}')
        print(f'D={np.rad2deg(self.D)}')

        # DEBUG: To confirm star locations match after transformation
        # Generate plots of how well fitting conforms to real star positions
        if plot:
            azt, elt = self.transform(self.starlist['x'], self.starlist['y'], self.x0, self.y0, self.rl,
                                      self.theta, self.A, self.B, self.C, self.D)
            fig = plt.figure()
            ax1 = fig.add_subplot(121, projection='polar')
            ax1.set_theta_zero_location('N')
            ax1.set_theta_direction(-1)
            ax2 = fig.add_subplot(122)
            cmap=plt.get_cmap('tab20')
    
            xn = (self.starlist['x'] - self.x0) / self.rl
            yn = (self.starlist['y'] - self.y0) / self.rl
            rt = np.sqrt(xn**2 + yn**2)
    
            for i in range(len(self.starlist)):
                ax1.scatter(az0[i], np.cos(el0[i]), s=5, color=cmap(i%20))
                ax1.scatter(azt[i], np.cos(elt[i]), facecolors='none', edgecolors=cmap(i%20))
                ax2.scatter(rt[i], self.starlist['el'][i], color=cmap(i%20), label=self.starlist['Name'][i])
            ax1.set_xlabel(r'$X/R_L$')
            ax1.set_ylabel(r'$Y/R_L$')
            ax1.set_title('Alignment of Transformed Star Position\nwith True Projected Position')
            elgrid = np.arange(0., 90., 10.)
            azgrid = np.arange(0., 360., 30.)
            ax1.set_rgrids(radii=np.cos(np.deg2rad(elgrid)), labels=elgrid)
            ax1.set_thetagrids(angles=azgrid, labels=azgrid)
    
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='Projected True Position', markerfacecolor='k', markersize=5),
                               Line2D([0], [0], marker='o', color='w', label='Transformed CCD Position', markeredgecolor='k', markersize=5)]
            ax1.legend(handles=legend_elements, loc='upper left')
            
            ax2.set_xlabel('R')
            ax2.set_ylabel('Elevation (deg)')
            ax2.set_title('Lens Function')
            ax2.set_xlim([0., 1.])
            ax2.set_ylim([0., 90.])
            ax2.grid()
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


    def transform(self, xc, yc, x0, y0, rl, theta, A, B, C, D):
        """Transformation"""

        xn = (xc - x0) / rl
        yn = (yc - y0) / rl

        r = np.sqrt(xn**2 + yn**2)
        lam = A + B * r + C * r**2 + D * r**3

        #phi = theta * np.pi / 180.0 - np.arctan2(yn, xn)
        phi = np.deg2rad(theta) - np.arctan2(yn, xn) + np.pi
        phi[phi<0.] += 2*np.pi
        phi[phi>=2*np.pi] -= 2*np.pi
        #phi = phi % 2*np.pi

        return phi, lam

    def residuals(self, params, x, y, p1, l1):
        """Residuals"""

        x0, y0, rl, theta, C, D = params
        A = np.pi / 2.0
        B = -(np.pi / 2.0 + C + D)
        p2, l2 = self.transform(x, y, x0, y0, rl, theta, A, B, C, D)

        # Haversine Formulation
        dl = (l2 - l1)
        dp = (p2 - p1)
        a = np.sin(dl/2)**2 + np.sin(dp/2)**2 * np.cos(l1) * np.cos(l2)
        res = 2 * np.asin(np.sqrt(a))

        ## Horizontal Residual
        #res = np.sqrt(np.cos(l1)**2 + np.cos(l2)**2 -2*np.cos(l1)*np.cos(l2)*np.cos(p2-p1))

        return res


    def initial_params(self, x, y, az0, el0, imax, jmax):
        """Initial parameters"""

        # Use center of image and half of y distance for x0, y0, and rl
        x0 = imax/2.
        y0 = jmax/2.
        rl = min(x0, y0)

        # appriximate lense function with line
        A, B, C, D = [np.pi / 2, -np.pi / 2, 0.0, 0.0]

        # calculate transformation with initial tranlation and lens function params but no rotation
        az, el = self.transform(x, y, x0, y0, rl, 0.0, A, B, C, D)

        # calculate rotation by the average difference in azimuth
        theta = np.mean((az0-az) % (2*np.pi)) * 180./np.pi

        return [x0, y0, rl, theta, C, D]


    def elev2r(self, elev):

        #import cmath as cm

        # This function is calculating the r value for a given elevation, or
        #   the inverse of the lens function.  There should be a clever way
        #   to do this with root finding given the lens function is just a
        #   polynomial, but it's actually rather complicated due to imaginary
        #   roots and roots outside the 0-1 range.  For now, a brute force
        #   linear interpolation gives a consistently correct answer.

        el = np.deg2rad(elev)

        rg = np.linspace(0., 1., 100)
        lg = self.A + self.B * rg + self.C * rg**2 + self.D * rg**3

        r = np.interp(el, lg[::-1], rg[::-1])

#        Delta0 = (self.C**2 - 3 * self.D * self.B).astype(complex)
#        Delta1 = (2 * self.C**3 - 9 * self.D * self.C * self.B + 27 * self.D**2 * (self.A-el)).astype(complex)
#        Gamma = ((Delta1 + np.sqrt(Delta1**2 - 4 * Delta0**3)) / 2)**(1./3.)
#        Gamma1 = Gamma*(-1+np.sqrt(-3+0j))/2
#        Gamma2 = Gamma*(-1-np.sqrt(-3+0j))/2
#
#        Gamma0 = np.array([Gamma, Gamma1, Gamma2])
#        print(np.min(np.abs(Gamma0.imag), axis=-1))
#
#        Gamma[np.isnan(Gamma)]
#        r = -(self.C + Gamma2 + Delta0/Gamma2)/(3 * self.D)
#        print('GAMMA')
#        print(Gamma)
#        print(Gamma1)
#        print(Gamma2)
#        print(elev, r)

#        ## SOLVE THIS LATER TONIGHT
#
#        if np.isscalar(el):
#            roots = np.roots([self.D, self.C, self.B, self.A-el])
#            r = np.extract(np.logical_and(0.<=roots, roots<=1.), roots)
#        else:
#            r = list()
#            for e in el:
#                roots = np.roots([self.D, self.C, self.B, self.A-e])
#                r.append(np.extract(np.logical_and(0<=roots, roots<=1), roots))
#        print(r)

        return r


    def calculate_position_array(self, site_lat, site_lon, alt, imax, jmax):

        # az/el array
        xc, yc = np.meshgrid(np.arange(imax), np.arange(jmax))
        az, el = self.transform(xc, yc, self.x0, self.y0, self.rl, self.theta, self.A, self.B, self.C, self.D)


        # lat/lon array
        x, y, z = pm.geodetic2ecef(site_lat, site_lon, 0.)
        e, n, u = pm.aer2enu(az, el, 1., deg=False)
        vx, vy, vz = pm.enu2uvw(e, n, u, site_lat, site_lon)
    
        earth = pm.Ellipsoid.from_name('wgs84')
        a2 = (earth.semimajor_axis + alt*1000.)**2
        b2 = (earth.semimajor_axis + alt*1000.)**2
        c2 = (earth.semiminor_axis + alt*1000.)**2
    
        A = vx**2/a2 + vy**2/b2 + vz**2/c2
        B = x*vx/a2 + y*vy/b2 + z*vz/c2
        C = x**2/a2 + y**2/b2 + z**2/c2 -1
    
        alpha = (np.sqrt(B**2-A*C)-B)/A
    
        lat, lon, alt = pm.ecef2geodetic(x + alpha*vx, y + alpha*vy, z + alpha*vz)

        return az, el, lat, lon

    def checkcal(self, image, site_lat):

        # Display image with stars
        fig, ax = plt.subplots()
        # Display image
        ax.imshow(image, cmap='gray')

        # Plot Zenith
        ax.scatter(self.x0, self.y0, s=50, color='red', marker='P', label='zenith')

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
        x1 = self.x0 - self.rl * np.cos(np.deg2rad(self.theta))
        y1 = self.y0 - self.rl * np.sin(np.deg2rad(self.theta))
        ax.plot([self.x0, x1], [self.y0, y1], color='k', linestyle=':', label='North')

        # Plot Polaris
        r0 = self.elev2r(site_lat)
        x = self.x0 - r0 * self.rl * np.cos(np.deg2rad(self.theta))
        y = self.y0 - r0 * self.rl * np.sin(np.deg2rad(self.theta))
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








