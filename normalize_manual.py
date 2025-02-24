#!/usr/bin/env python3.12
#
# Quick and very dirty code for continuum normalisation. Experimental branch
# Author: Eugene Semenko
# Last modification: 24 Feb 2025
# russia delenda est. Слава Україні!


from sys import exit, modules
import os.path
import itertools
import numpy as np
from astropy.io import fits
from scipy.interpolate import Akima1DInterpolator, make_smoothing_spline
from numpy.polynomial.legendre import legfit, legval
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy.optimize import curve_fit, leastsq
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
from matplotlib.widgets import Button, Slider, Cursor, TextBox, SpanSelector, CheckButtons
import warnings


warnings.filterwarnings("ignore")


fontsize = 10
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['font.family'] = 'sans-serif'

hoverclr = "#ffffe4"

# mpl.use('TkAgg')
# mpl.rcParams['font.serif'] = 'Times Palatino, New Century Schoolbook, Bookman,  Computer Modern Roman'

c = 299792.458 # speed of light in km/s

# General functions for reading IRAF FITS
def read_multispec(input_file):
    """
    This function reads the input file and returns wavelength and flux.
    Recognize IRAF multipspec spectra with different types of dispersion solution
    ES. 2020-10-21
    """
    try:
        hdu = fits.open(input_file)
    except Exception:
        print("Error while opening the input file")
    finally:
        header = hdu[0].header
        spectrum = hdu[0].data
        hdu.close()
    sizes = np.shape(spectrum)
    if len(sizes) == 1:
        nspec = 1
        npix = sizes[0]
    elif len(sizes) == 2:
        nspec = sizes[0]
        npix = sizes[1]
    elif len(sizes) >=3:
        nspec = sizes[-2]
        npix = sizes[-1]
        spectrum = spectrum[0]
    waves = np.zeros((nspec, npix), dtype=float)
    # try to recognize the type of dispersion
    if 'CTYPE1' in header:
        if header['CTYPE1'].strip() == 'LINEAR': # Linear dispersion solution
            crpix1 = header['CRPIX1']
            crval1 = header['CRVAL1']
            cd1_1 = header['CD1_1']
            wave = (np.arange(npix, dtype=float) + 1 - crpix1) * cd1_1 + crval1
            for i in range(nspec):
                waves[i, :] = wave
            if 'DC-FLAG' in header:
                if header['DC-FLAG'] == 1:
                    waves = 10**waves
        elif header['CTYPE1'].strip() == 'MULTISPE': # IRAF multispec data
            try:
                wat2 = header['WAT2_*']
            except Exception:
                print("Header does not contain keywords required for multispec data")
            finally:
                count_keys = len(wat2)
            long_wat2 = ""
            wave_params = np.zeros((nspec, 24), dtype=float)
            for i in wat2:
                key = header[i].replace('\'', '')
                if len(key) < 68: key += ' '
                long_wat2 += key
            for i in range(nspec):
                idx_b = long_wat2.find("\"", long_wat2.find("spec"+str(i+1)+" ="), -1)
                idx_e = long_wat2.find("\"", idx_b+1, -1)
                temparr = np.asarray(long_wat2[idx_b+1:idx_e].split())
                wave_params[i, 0:len(temparr)] = temparr
                if wave_params[i, 2] == 0 or wave_params[i, 2] == 1:
                    waves[i, :] = np.arange(npix, dtype=float) * wave_params[i, 4] \
                                + wave_params[i, 3]
                    if wave_params[i, 2] == 1:
                        waves[i, :] = 10**waves[i, :]
                else: # Non-linear solution. Not tested
                    waves[i, :] = nonlinearwave(npix, long_wat2[idx_b+1:idx_e])
        elif header['CTYPE1'].strip() == 'PIXEL':
            waves[:,:] = np.arange(npix)+1
    return waves,spectrum,header

def nonlinearwave(nwave, specstr):
    """
    This function is a modified version of the corresponding unit from readmultispec.py
    (https://raw.githubusercontent.com/kgullikson88/General/master/readmultispec.py)
    Eugene Semenko, 2020-10-21

    Compute non-linear wavelengths from multispec string
    Returns wavelength array and dispersion fields.
    Raises a ValueError if it can't understand the dispersion string.
    """

    fields = specstr.split()
    if int(fields[2]) != 2:
        raise ValueError('Not nonlinear dispersion: dtype=' + fields[2])
    if len(fields) < 12:
        raise ValueError('Bad spectrum format (only %d fields)' % len(fields))
    wt = float(fields[9])
    w0 = float(fields[10])
    ftype = int(fields[11])
    if ftype == 3:
        # cubic spline
        if len(fields) < 15:
            raise ValueError('Bad spline format (only %d fields)' % len(fields))
        npieces = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])
        if len(fields) != 15 + npieces + 3:
            raise ValueError('Bad order-%d spline format (%d fields)' % (npieces, len(fields)))
        coeff = np.asarray(fields[15:], dtype=float)
        # normalized x coordinates
        s = (np.arange(nwave, dtype=float) + 1 - pmin) / (pmax - pmin) * npieces
        j = s.astype(int).clip(0, npieces - 1)
        a = (j + 1) - s
        b = s - j
        x0 = a ** 3
        x1 = 1 + 3 * a * (1 + a * b)
        x2 = 1 + 3 * b * (1 + a * b)
        x3 = b ** 3
        wave = coeff[j] * x0 + coeff[j + 1] * x1 + coeff[j + 2] * x2 + coeff[j + 3] * x3
    elif ftype == 1 or ftype == 2:
        # chebyshev or legendre polynomial
        # legendre not tested yet
        if len(fields) < 15:
            raise ValueError('Bad polynomial format (only %d fields)' % len(fields))
        order = int(fields[12])
        pmin = float(fields[13])
        pmax = float(fields[14])
        if len(fields) != 15 + order:
            # raise ValueError('Bad order-%d polynomial format (%d fields)' % (order, len(fields)))
            order = len(fields) - 15
        coeff = np.asarray(fields[15:], dtype=float)
        # normalized x coordinates
        pmiddle = (pmax + pmin) / 2
        prange = pmax - pmin
        x = (np.arange(nwave, dtype=float) + 1 - pmiddle) / (prange / 2)
        p0 = np.ones(nwave, dtype=float)
        p1 = x
        wave = p0 * coeff[0] + p1 * coeff[1]
        for i in range(2, order):
            if ftype == 1:
                # chebyshev
                p2 = 2 * x * p1 - p0
            else:
                # legendre
                p2 = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i
            wave = wave + p2 * coeff[i]
            p0 = p1
            p1 = p2
    else:
        raise ValueError('Cannot handle dispersion function of type %d' % ftype)
    return wave
##########

# Classes
class Spec():
	def __init__(self, file_iraf):
		try:
			wl, fl, hdr = read_multispec(file_iraf)
		except Exception as e:
			print(f"{e}")
			exit(1)
		# Below are the class attributes
		self.norders = wl.shape[0]  # Number of orders
		self.w0 = wl[0][0]          # The short end of the spectrum
		self.wn = wl[-1][-1]        # The long end of the spectrum
		self.w = wl
		self.f = fl
		self.header = hdr
		self.fname = file_iraf
		self.activeOrder = -1
		units = str(self.w0).find('.')
		if units == 4:
			self.u = 'Å'
		elif units == 3:
			self.u = 'nm'
		print(f"File \'{file_iraf}\' has been successfully read")
		print(f"Spectrum of {hdr['OBJNAME']} spans from {self.w0:.3f} to {self.wn:.3f} {self.u} in {self.norders} orders")
		return None

class RunApplication(object):
	def __init__(self, Sp):	# Initialize graphics using Matplotlib
		self.data = Sp
		self.data.cont = np.ones(self.data.w.shape)
		self.data.norm = np.ones(self.data.w.shape)
		self.fig = plt.figure(figsize=(15,8))
		self.ax1 = self.fig.add_subplot(2,1,1)
		self.ax1.set_ylabel('Intensity')
		self.ax1.set_xlabel(f"Wavelength ({Sp.u})")
		self.ax2 = self.fig.add_subplot(2,1,2)
		self.ax2.set_xlabel(f"Wavelength ({Sp.u})")
		self.ax2.set_ylabel('Normalised intensity')
		plt.subplots_adjust(bottom=0.16, top=0.92, left=0.08, right=0.97, hspace=0.6)

		# Defaults
		self.curve_main = None
		self.curve_prev = None
		self.curve_next = None
		self.selected = None

		# Controls
		# Slider "Active order"
		axis_select = plt.axes([0.08, 0.95, 0.89, 0.03])
		self.slider_select = Slider(axis_select, 'Active order', valmin=0, valmax=Sp.norders, valinit=0, valstep=1, color="green")
		self.slider_select.on_changed(self.redraw_order)

		# Button "Full range"
		axis_range = plt.axes([0.08, 0.53, 0.1, 0.03])
		self.button_range = Button(axis_range, 'Full range', color='black', hovercolor='black', useblit=True)
		# Button "Add selection"
		axis_region = plt.axes([0.20, 0.53, 0.1, 0.03])
		self.button_region = Button(axis_region, 'Add selection', color='black', hovercolor='black', useblit=True)
		# Button "Add special"
		axis_addpoint = plt.axes([0.32, 0.53, 0.1, 0.03])
		self.button_addpoint = Button(axis_addpoint, 'Add special', color='black', hovercolor='black', useblit=True)
		# Button "Picasso"
		axis_picasso = plt.axes([0.44, 0.53, 0.1, 0.03])
		self.button_picasso = Button(axis_picasso, "¡Soy Picasso!", color='black', hovercolor='black', useblit=True)

		# Checkbuton widgets
		axCheckButton_extra = plt.axes([0.56, 0.53, 0.11, 0.03])
		self.chkbox_extra = CheckButtons(axCheckButton_extra, labels=['Extra orders'], actives=[True], label_props=({'size': ['large']}))
		self.chkbox_extra.on_clicked(self.showExtra)
		self.showE = True # Don't show extra orders
		axCheckButton_units = plt.axes([0.69, 0.53, 0.16, 0.03])
		self.chkbox_units = CheckButtons(axCheckButton_units, ['Use pixels'], [False], label_props=({'size': ['large']}))
		self.chkbox_units.on_clicked(self.changeUnits)
		self.unitsP = False  # Angstroms/nm by default

		# Button "Erase selection"
		axis_eraseregion = plt.axes([0.87, 0.53, 0.1, 0.03])
		self.button_eraseregion = Button(axis_eraseregion, 'Erase selection', color='black', hovercolor='black', useblit=True)

		# Button "Select function"
		axis_func = plt.axes([0.08, 0.48, 0.1, 0.03])
		self.button_func = Button(axis_func, 'Function', color='black', hovercolor='black') # #d8dcd6
		# Text boxes
		axis_order = plt.axes([0.22, 0.48, 0.06, 0.03])
		self.orderbox = TextBox(axis_order, label="Order ", initial="4", textalignment="center", color='#d8dcd6')
		axis_iter = plt.axes([0.32, 0.48, 0.06, 0.03])
		self.iterbox = TextBox(axis_iter, label="N_iter ", initial="5", textalignment="center", color='#d8dcd6')
		axis_siglow = plt.axes([0.42, 0.48, 0.06, 0.03])
		self.siglowbox = TextBox(axis_siglow, label="σ_low ", initial="1.1", textalignment="center", color='#d8dcd6')
		axis_sighigh = plt.axes([0.52, 0.48, 0.06, 0.03])
		self.sighighbox = TextBox(axis_sighigh, label="σ_high ", initial="6.0", textalignment="center", color='#d8dcd6')

		axis_output_norm = plt.axes([0.08, 0.07, 0.4, 0.03])
		self.norm_filename = TextBox(axis_output_norm, label="Normal.", initial="norm_"+self.data.fname, textalignment="left", color='#d8dcd6')
		axis_output_cont = plt.axes([0.55, 0.07, 0.4, 0.03])
		self.cont_filename = TextBox(axis_output_cont, label="Cont.", initial="cont_"+self.data.fname, textalignment="left", color='#d8dcd6')
		axis_output_merged = plt.axes([0.318, 0.025, 0.4, 0.03])
		self.merged_filename = TextBox(axis_output_merged, label="→", initial="comb_"+self.data.fname, textalignment="left", color='#d8dcd6')

		# Button "Try"
		axis_try = plt.axes([0.63, 0.48, 0.1, 0.03])
		self.button_try = Button(axis_try, 'Try fitting!', color='black', hovercolor='black', useblit=True) #feb308
		self.button_try.on_clicked(self.fitterEval)
		# Button "Normalise"
		axis_norm = plt.axes([0.75, 0.48, 0.1, 0.03])
		self.button_norm = Button(axis_norm, 'Normalise', color='black', hovercolor='black', useblit=True)  #9be5aa
		self.button_norm.on_clicked(self.normaliseSpec)
		# Button "Discard fit"
		axis_discard = plt.axes([0.87, 0.48, 0.1, 0.03])
		self.button_discard = Button(axis_discard, 'Discard fit', color='black', hovercolor='black', useblit=True)  #c85a53
		self.button_discard.on_clicked(self.discardFit)

		# Button "Save result"
		axis_save = plt.axes([0.08, 0.025, 0.1, 0.03])
		self.button_save = Button(axis_save, 'Save results ↑↗', color='black', hovercolor='black', useblit=True)  #c5c9c7
		self.button_save.on_clicked(self.saveOutput)
		# Button "Merge"
		axis_merge = plt.axes([0.20, 0.025, 0.1, 0.03])
		self.button_merge = Button(axis_merge, 'Merge orders', color='black', hovercolor='black', useblit=True)  #c5c9c7
		self.button_merge.on_clicked(self.stitchOutput)
		# Button "Exit"
		axis_exit = plt.axes([0.85, 0.025, 0.1, 0.03])
		self.button_exit = Button(axis_exit, 'Exit app.', color='#ff474c', hovercolor=hoverclr, useblit=True)
		self.button_exit.on_clicked(self.exit)

		self.initialize_plot()
		cursor = Cursor(self.ax1, useblit=True, color='red', linewidth=0.5)
		plt.show()
		return None

	def exit(self, event):
		exit(0)

	def initialize_plot(self):
		self.ax1.cla()
		for i in range(self.data.norders):
			self.ax1.plot(self.data.w[i], self.data.f[i], lw=0.5)
		self.ax1.set_xlim(self.data.w0-0.02*self.data.w0, self.data.wn+0.02*self.data.wn)
		self.ax2.set_xlim(self.data.w0-0.02*self.data.w0, self.data.wn+0.02*self.data.wn)
		self.fig.canvas.draw_idle()
		self.cid = None
		self.curfit = None
		self.curnorm = None
		return None

	def redraw_order(self, event):
		self.ax1.cla()
		self.fig.canvas.flush_events()
		self.fig.canvas.mpl_disconnect(self.cid)
		self.fitterFunctions = itertools.cycle(['Chebyshev', 'Legendre', 'Spline'])
		self.data.cur_wcont = np.array([])
		self.data.cur_cont = np.array([])
		if self.curfit != None:
			self.curfit[0].remove()
			self.curfit = None
		if self.curnorm != None:
			self.curnorm[0].remove()
			self.curnorm = None
		if self.slider_select.val > 0:
			self.main_segment = np.array([])
			self.main_segment_w = np.array([])
			self.aux_segment = np.array([])
			self.aux_segment_w = np.array([])
			self.data.activeOrder = self.slider_select.val
			if self.data.activeOrder > 1 and self.showE:
				if self.unitsP:
					self.curve_prev = self.ax1.plot(np.arange(len(self.data.w[self.data.activeOrder-2])), self.data.f[self.data.activeOrder-2], ls='-', lw=0.8, color="#929591")
				else:
					self.curve_prev = self.ax1.plot(self.data.w[self.data.activeOrder-2], self.data.f[self.data.activeOrder-2], ls='-', lw=0.8, color="#929591")
					self.aux_segment_w = np.append(self.aux_segment_w, self.data.w[self.data.activeOrder-2])
					self.aux_segment = np.append(self.aux_segment, self.data.f[self.data.activeOrder-2])
			if self.data.activeOrder < self.data.norders and self.showE:
				if self.unitsP:
					self.curve_next = self.ax1.plot(np.arange(len(self.data.w[self.data.activeOrder])), self.data.f[self.data.activeOrder], ls='-', lw=0.8, color="#59656d")
				else:
					self.curve_next = self.ax1.plot(self.data.w[self.data.activeOrder], self.data.f[self.data.activeOrder], ls='-', lw=0.8, color="#929591")
					self.aux_segment_w = np.append(self.aux_segment_w, self.data.w[self.data.activeOrder])
					self.aux_segment = np.append(self.aux_segment, self.data.f[self.data.activeOrder])
			if self.unitsP:
				self.curve_main = self.ax1.plot(np.arange(len(self.data.w[self.data.activeOrder-1])), self.data.f[self.data.activeOrder-1], ls='-', lw=0.8, color="#030aa7")
				self.main_segment_w = np.append(self.main_segment_w, np.arange(len(self.data.w[self.data.activeOrder-1])))
			else:
				self.curve_main = self.ax1.plot(self.data.w[self.data.activeOrder-1], self.data.f[self.data.activeOrder-1], ls='-', lw=0.8, color="#030aa7")
				self.main_segment_w = np.append(self.main_segment_w, self.data.w[self.data.activeOrder-1])
			self.main_segment = np.append(self.main_segment, self.data.f[self.data.activeOrder-1])
			if not self.unitsP:
				self.aux_segment_w = self.aux_segment_w.flatten()
				self.aux_segment = self.aux_segment.flatten()
			if self.button_range.color == "black":
				self.activateIt(self.button_range, "#75bbfd")
				self.button_range.on_clicked(self.selectAll)
			if self.button_region.color == "black":
				self.activateIt(self.button_region, "#75bbfd")
				self.button_region.on_clicked(self.addRegion)
			if self.button_addpoint.color == "black":
				self.activateIt(self.button_addpoint, "#75bbfd")
				self.button_addpoint.on_clicked(self.addSpecial)
			if self.button_picasso.color == "black":
				self.activateIt(self.button_picasso, "#ffd1df")
				self.button_picasso.on_clicked(self.soyPicasso)
			self.make_inactive([self.button_try, self.button_norm, self.button_discard, self.button_eraseregion])
		else:
			for i in range(self.data.norders):
				self.ax1.plot(self.data.w[i], self.data.f[i], lw=0.5)
			self.ax1.set_xlim(self.data.w0-0.02*self.data.w0, self.data.wn+0.02*self.data.wn)
			self.ax2.set_xlim(self.data.w0-0.02*self.data.w0, self.data.wn+0.02*self.data.wn)
			self.make_inactive([self.button_range, self.button_region, self.button_addpoint, self.button_picasso, self.button_func, self.button_try, self.button_norm, self.button_discard])
		if self.unitsP:
			self.ax1.set_xlabel("X Coordinate (pix)")
		else:
			self.ax1.set_xlabel(f"Wavelength ({self.data.u})")
		self.fig.canvas.draw_idle()
		return None

	def selectAll(self, event):
		self.fig.canvas.mpl_disconnect(self.cid)
		self.data.cur_wcont = np.append(self.data.cur_wcont, [self.main_segment_w, self.aux_segment_w]).flatten()
		self.data.cur_cont = np.append(self.data.cur_cont, [self.main_segment, self.aux_segment]).flatten()
		idx = np.argsort(data.cur_wcont)
		self.data.cur_wcont = self.data.cur_wcont[idx]
		self.data.cur_cont = self.data.cur_cont[idx]
		self.ax1.plot(self.data.cur_wcont, self.data.cur_cont, color="#be03fd", lw=1.2, ls='-')
		self.fig.canvas.draw_idle()
		if self.button_try.color == "black":
			self.activateIt(self.button_try, "#feb308")
		if self.button_eraseregion.color == "black":
			self.activateIt(self.button_eraseregion, "#c85a53")
			self.button_eraseregion.on_clicked(self.redraw_order)
		if self.button_func.color == "black":
			self.activateIt(self.button_func, "#d8dcd6")
			self.button_func.on_clicked(self.selectFunction)
		self.fig.canvas.draw_idle()
		return None

	def addRegion(self, event):
		def onselect(eclick, erelease):
			if eclick < erelease:
				x1 = eclick; x2 = erelease
			else:
				x1 = erelease; x2 = eclick
			if np.any((x1 >= self.main_segment_w[0]) & (x1 <= self.main_segment_w[-1])) or np.any((x2 >= self.main_segment_w[0]) & (x2 <= self.main_segment_w[-1])):
				idx = np.where((self.main_segment_w >= x1) & (self.main_segment_w <= x2))[0]
				self.data.cur_wcont = np.append(self.data.cur_wcont, self.main_segment_w[idx])
				self.data.cur_cont = np.append(self.data.cur_cont, self.main_segment[idx])
				self.ax1.plot(self.main_segment_w[idx], self.main_segment[idx], color="#be03fd", lw=1.2, ls='-')
			else:
				idx = np.where((self.aux_segment_w >= x1) & (self.aux_segment_w <= x2))[0]
				self.data.cur_wcont = np.append(self.data.cur_wcont, self.aux_segment_w[idx])
				self.data.cur_cont = np.append(self.data.cur_cont, self.aux_segment[idx])
				self.ax1.plot(self.aux_segment_w[idx], self.aux_segment[idx], color="#be03fd", lw=1.2, ls='-')
			self.fig.canvas.draw_idle()
			return None
		self.fig.canvas.mpl_disconnect(self.cid)
		props = dict(facecolor='blue', alpha=0.5)
		spansel = SpanSelector(self.ax1, onselect, "horizontal", useblit=True, props=dict(alpha=0.5, facecolor="tab:blue"), interactive=False, drag_from_anywhere=True)
		self.cid = self.fig.canvas.mpl_connect('resize_event', spansel)
		if self.button_func.label.get_text() != "Function" and self.button_try.color == "black":
			self.activateIt(self.button_try, "#feb308")
		if self.button_eraseregion.color == "black":
			self.activateIt(self.button_eraseregion, "#c85a53")
			self.button_eraseregion.on_clicked(self.redraw_order)
		if self.button_func.color == "black":
			self.activateIt(self.button_func, "#d8dcd6")
			self.button_func.on_clicked(self.selectFunction)
		self.fig.canvas.draw_idle()
		return None

	def addSpecial(self, event):
		def onclick(event):
			if event.button==1 and event.inaxes in [self.ax1]:
				self.data.cur_wcont = np.append(self.data.cur_wcont, np.repeat(event.xdata, 5))
				self.data.cur_cont = np.append(self.data.cur_cont, np.repeat(event.ydata, 5))
				self.ax1.plot(event.xdata, event.ydata, marker='o', ms=5, color='#e50000', mec='k', mew=0.7)
				self.fig.canvas.draw_idle()
				self.fig.canvas.mpl_disconnect(self.cid)
			return None
		self.fig.canvas.mpl_disconnect(self.cid)
		self.cid = self.fig.canvas.mpl_connect('button_press_event', onclick)
		if self.button_eraseregion.color == "black":
			self.activateIt(self.button_eraseregion, "#c85a53")
			self.button_eraseregion.on_clicked(self.redraw_order)
		if self.button_func.label.get_text() != "Function" and self.button_try.color == "black":
			self.activateIt(self.button_try, "#feb308")
		if self.button_func.color == "black":
			self.activateIt(self.button_func, "#d8dcd6")
			self.button_func.on_clicked(self.selectFunction)
		self.fig.canvas.draw_idle()
		return None

	def soyPicasso(self, event):
		def onclick(event):
			if event.button==1 and event.inaxes in [self.ax1]:
				self.data.cur_wcont = np.append(self.data.cur_wcont, event.xdata)
				self.data.cur_cont = np.append(self.data.cur_cont, event.ydata)
				self.ax1.plot(event.xdata, event.ydata, marker='o', ms=5, color='#e50000', mec='k', mew=0.7)
				self.fig.canvas.draw_idle()
			elif event.button > 1:
				self.fig.canvas.mpl_disconnect(self.cid)
			return None
		self.fig.canvas.mpl_disconnect(self.cid)
		self.redraw_order(None)
		if self.button_eraseregion.color == "black":
			self.activateIt(self.button_eraseregion, "#c85a53")
			self.button_eraseregion.on_clicked(self.redraw_order)
		self.make_inactive([self.button_range, self.button_region, self.button_addpoint])
		self.iterbox.set_val("0")
		self.fitterFunc = "Spline"
		self.button_func.label.set_text(self.fitterFunc)
		self.data.cur_wcont = np.array([])
		self.data.cur_cont = np.array([])
		self.cid = self.fig.canvas.mpl_connect('button_press_event', onclick)
		if self.button_func.color == "black":
			self.activateIt(self.button_func, "#d8dcd6")
			self.button_func.on_clicked(self.selectFunction)
		if self.button_func.label.get_text() != "Function" and self.button_try.color == "black":
			self.activateIt(self.button_try, "#feb308")
		self.fig.canvas.draw_idle()
		return None

	def selectFunction(self, event):
		if self.button_func.label.get_text() != "Function" or self.button_try.color == "black":
			self.activateIt(self.button_try, "#feb308")
		self.fitterFunc = next(self.fitterFunctions)
		if self.fitterFunc != 'Spline':
			self.iterbox.set_val("5")
		else:
			self.iterbox.set_val("0")
		self.button_func.label.set_text(self.fitterFunc)
		return None

	def changeUnits(self, event):
		self.unitsP = True if self.chkbox_units.get_status()[0] else False
		self.redraw_order(event)
		return None

	def showExtra(self, event):
		self.showE = True if self.chkbox_extra.get_status()[0] else False
		self.redraw_order(event)
		return None

#  Fitter
	def fitterEval(self, event):
		def reject_points(w, f, cont, low_rej, high_rej, func, order):
			resid = f - cont
			stdr = np.std(resid)
			idx = np.where((resid >= -low_rej * stdr) & (resid <= high_rej * stdr))
			coef = fit_poly(w[idx], f[idx], func, order)
			return w[idx], f[idx], cont, coef

		def fit_poly(w, r, func, order):
			if func == "Legendre":
				coef = legfit(w, r, order)
			elif func == "Chebyshev":
				coef = chebfit(w, r, order)
			elif func == "Spline":
				# coef = Akima1DInterpolator(w, r, method="makima", extrapolate=True)
				coef = make_smoothing_spline(w, r, lam=order)
			return coef

		def fit_cont(w, func, coef):
			if func == "Legendre":
				return legval(w, coef)
			elif func == "Chebyshev":
				return chebval(w, coef)
			elif func == "Spline":
				return coef(w)

		def normaliseIraf(w2fit, w_0, r_0, fit_func, fit_ord, fit_niter, fit_low_rej, fit_high_rej):
			w_init = w_0.copy()
			r_init = r_0.copy()
			cont_lev = np.zeros(len(r_init))
			if fit_niter <= 1:
				fit_niter = fit_niter + 2
			w_tmp = w_init
			r_tmp = r_init
			idx = np.argsort(w_tmp)
			w_tmp = w_tmp[idx]
			r_tmp = r_tmp[idx]
			for j in range(fit_niter-1):
				coef = fit_poly(w_tmp, r_tmp, fit_func, fit_ord)
				cont = fit_cont(w_tmp, fit_func, coef)
				w_tmp, r_tmp, cont, coef = reject_points(w_tmp, r_tmp, cont, fit_low_rej, fit_high_rej, fit_func, fit_ord)
				cont_full = fit_cont(w_init, fit_func, coef)
			cont_lev = fit_cont(w2fit, fit_func, coef)
			idx_wrong = np.where(cont_lev == 0.)[0]
			cont_lev[idx_wrong] = r_init[idx_wrong]
			return cont_full, cont_lev
		if self.button_try.color != "black":
			self.data.cur_wcont = self.data.cur_wcont.flatten()
			self.data.cur_cont = self.data.cur_cont.flatten()
			idx = np.argsort(self.data.cur_wcont)
			self.data.cur_wcont = self.data.cur_wcont[idx]
			self.data.cur_cont = self.data.cur_cont[idx]
			cont_full, self.cur_cont_fit = normaliseIraf(self.main_segment_w, self.data.cur_wcont, self.data.cur_cont, self.fitterFunc, int(self.orderbox.text), int(self.iterbox.text), float(self.siglowbox.text), float(self.sighighbox.text))
			if self.curfit != None:
				self.curfit[0].remove()
				self.curnorm[0].remove()
				self.curfit = None
				self.curnorm = None
			self.curfit = self.ax1.plot(self.data.cur_wcont, cont_full, ls='-', color='red', lw=1)
			self.curnorm = self.ax2.plot(self.main_segment_w, self.main_segment/self.cur_cont_fit, ls='-', color='#02ab2e', lw=1)
			if len(self.aux_segment_w) != 0:
				self.ax2.set_xlim(np.min([self.aux_segment_w[0], self.main_segment_w[0]]), np.max([self.aux_segment_w[-1], self.main_segment_w[-1]]))
			else:
				self.ax2.set_xlim(self.main_segment_w[0], self.main_segment_w[-1])
			if self.button_norm.color == "black":
				self.activateIt(self.button_norm, "#9be5aa")
			if self.button_discard.color == "black":
				self.activateIt(self.button_discard, "#c85a53")
			self.fig.canvas.draw_idle()
			self.fig.canvas.flush_events()
		return None
# End of

	def make_inactive(self, elems):
		for el in elems:
			el.color = "black"
			el.hovercolor = "black"
		return None

	def activateIt(self, elem, clr):
		elem.color = clr
		elem.hovercolor = hoverclr
		return None

	def discardFit(self, event):
		if self.curfit != None and self.button_discard.color != "black":
			self.curfit[0].remove()
			self.curnorm[0].remove()
			self.curfit = None
			self.curnorm = None
			self.fig.canvas.draw_idle()
		return None

	def normaliseSpec(self, event):
		if self.curfit != None and self.button_norm.color != "black":
			self.data.norm[self.data.activeOrder-1] = (self.data.f[self.data.activeOrder-1]/self.cur_cont_fit).copy()
			self.data.cont[self.data.activeOrder-1] = self.cur_cont_fit.copy()
			self.ax2.plot(self.data.w[self.data.activeOrder-1], self.data.norm[self.data.activeOrder-1], ls='-', color='#607c8e', lw=0.5)
			self.curnorm[0].remove()
			self.curnorm = None
			self.curfit[0].remove()
			self.curfit = None
			self.redraw_order(None)
			if self.button_save.color == "black":
				self.activateIt(self.button_save, "#c5c9c7")
		return None

	def saveOutput(self, event):
		clobber = False
		if self.button_save.color != "black":
			hdu_n = fits.PrimaryHDU(self.data.norm)
			hdu_n.header = self.data.header.copy()
			hdu_c = fits.PrimaryHDU(self.data.cont)
			hdu_c.header = self.data.header.copy()
			hdu_c.header['OBJNAME'] = "Continuum"
			norm_out_name = self.norm_filename.text.strip()
			cont_out_name = self.cont_filename.text.strip()
			if os.path.isfile(norm_out_name):
				print(f"File {norm_out_name} exists. Do you want to overwrite it? ")
				ans = input("(y/n) -> ").strip()
				if ans[0].lower() == "y":
					clobber = True
			hdu_n.writeto(norm_out_name, overwrite=clobber)
			clobber = False
			if os.path.isfile(cont_out_name):
				print(f"File {cont_out_name} exists. Do you want to overwrite it? ")
				ans = input("(y/n) -> ").strip()
				if ans[0].lower() == "y":
					clobber = True
			hdu_c.writeto(cont_out_name, overwrite=clobber)
			clobber = False
			if os.path.isfile(cont_out_name) and os.path.isfile(norm_out_name):
				print("Output files have been saved successfully")
			else:
				print("Warning: Not all output files have been saved")
			self.fig.canvas.flush_events()
			self.activateIt(self.button_merge, "#c5c9c7")
			self.fig.canvas.draw_idle()
		return None


	def stitchOutput(self, event):
		clobber = False
		if self.button_merge.color != "black":
			w = self.data.w.flatten()
			f = self.data.f.flatten()
			c = self.data.cont.flatten()
			dw = []
			gaps = []
			for i in range(1, self.data.norders):
				dwi = np.mean(np.diff(self.data.w[i-1]))
				dw.append(dwi)
				if self.data.w[i][0] >= self.data.w[i-1][-1]+dwi:
					gaps.append([self.data.w[i-1][-1], self.data.w[i][0]])
			idx = np.argsort(w)
			w = w[idx]; f = f[idx]; c = c[idx]
			dw = np.mean(dw)
			print(f"Mean dl = {dw:.3f}")
			w_out = np.arange(w[0], w[-1]+dw, dw)
			if args.interpolator == "spectres" and "spectres" in modules:
				f_int = spectres(w_out, w, f)
				c_int = spectres(w_out, w, c)
			elif args.interpolator == "spline":
				spl_flux = make_smoothing_spline(w, f, lam=1e-4)
				spl_continuum = make_smoothing_spline(w, c, lam=1e-4)
				f_int = spl_flux(w_out)
				c_int = spl_continuum(w_out)
			f_out =  f_int/c_int
			if len(gaps) != 0:
				for g in gaps:
					idx = np.where((w_out >= g[0]) & (w_out <= g[1]))
					f_out[idx] = 1.0
			self.ax2.plot(w_out, f_out, ls='-', color="#0504aa", lw=0.6)
			self.fig.canvas.draw_idle()
			stitched_fname = self.merged_filename.text.strip()
			if os.path.isfile(stitched_fname):
				print(f"File {stitched_fname} exists. Do you want to overwrite it? ")
				ans = input("(y/n) -> ").strip()
				if ans[0].lower() != "y":
					print("OK. Skipping")
					return None
			np.savetxt(stitched_fname, np.vstack((w_out, f_out)).transpose(), fmt="%.4f")
			if os.path.isfile(stitched_fname):
				print("Output file with stitched spectrum has been saved successfully")
			else:
				print("Warning: Output file with stitched spectrum has not been saved")
		return None


# Main block
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("spec", help="Input spectrum", type=str, default="")
	parser.add_argument("--interpolator", help="Choose the interpolating algorithm [spectres, spline]", type=str, default="spline")
	args = parser.parse_args()
	A = Spec(args.spec.strip())
	cnv = RunApplication(A)
	# plt.show()
	exit(0)
