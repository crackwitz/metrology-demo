#!/usr/bin/env python3

# written in 2020 by Christoph Rackwitz <christoph.rackwitz@gmail.com>
# use only for war purposes and sarcasm

import sys
import numpy as np
import cv2 as cv
import scipy.interpolate # interp2d
import scipy.ndimage # gaussian_filter1d

def build_transform(p0, p1, stride=None, nsamples=None):
	"builds an affine transform with x+ along defined line"
	# use one of stride (in pixels) or nsamples (absolute value)

	(x0, y0) = p0
	(x1, y1) = p1

	dx = x1 - x0
	dy = y1 - y0

	length = np.hypot(dx, dy)

	if nsamples is not None:
		#stride = length / nsamples
		factor = 1 / nsamples

	else:
		if stride is None:
			stride = 1.0

		factor = stride / length
		nsamples = int(round(length / stride))

	# map: src <- dst (use WARP_INVERSE_MAP flag for warpAffine)
	H = np.eye(3, dtype=np.float64) # homography

	H[0:2, 0] = (dx, dy) # x unit vector
	H[0:2, 1] = (-dy, dx) # y unit vector is x rotated by 90 degrees

	H[0:2, 0:2] *= factor

	H[0:2, 2] = (x0, y0) # translate onto starting point

	# take affine part of homography
	assert np.isclose(a=H[2], b=(0,0,1)).all() # we didn't touch those but let's better check
	A = H[0:2, :]

	return (nsamples, A)

def sample_opencv(im, M, nsamples):
	
	# use transform to get samples
	samples = cv.warpAffine(im, M=M, dsize=(nsamples, 1), flags=cv.WARP_INVERSE_MAP | cv.INTER_CUBIC)

	# data is a row vector
	samples = samples[0]

	# INTER_CUBIC seems to break down beyond 1/32 sampling (discretizes).
	# there might be fixed point algorithms at work

	return samples

def sample_scipy(im, M, nsamples):
	# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html

	coords = np.vstack([np.arange(nsamples), np.zeros(nsamples), np.ones(nsamples)])

	coords_mapped = M.astype(np.float32) @ coords # @ = np.dot

	# FIXME: interp2d() is an expensive operation if the image is large
	#        maybe crop to bounding box of line (bbox of coords_mapped)?
	sampler = scipy.interpolate.interp2d(x=np.arange(imw), y=np.arange(imh), z=im, kind='cubic')

	sampler = np.vectorize(sampler) # doesn't cake coordinate pairs as is, vectorize() handles that (!= execution speed!)
	samples = sampler(*coords_mapped) # fairly fast compared to building the sampler (interp2d)

	return samples


if __name__ == '__main__':
	do_display = True # see below
	do_invert = True

	# to remove pixel noise
	smoothing_sigma = 2 # in pixels

	# define a line segment to sample along
	p0, p1 = (1320, 2500), (1320, 2100)
	stride = 1/4 # sample stride in pixels

	# the picture to work with
	if len(sys.argv) >= 2:
		imfname = sys.argv[1]
	else:
		imfname = "dish-1.jpg"

	########## here be dragons ##########

	decimals = max(0, int(np.ceil(-np.log10(stride))))

	print("loading picture...", end=" ", flush=True)
	im = cv.imread(imfname, cv.IMREAD_GRAYSCALE)
	imh, imw = im.shape[:2]
	if do_invert:
		im = 255-im # invert
	im = im.astype(np.float32)# * np.float32(1/255)
	print("done")

	# build transform
	nsamples, M = build_transform(p0, p1, stride=stride)

	print(f"taking {nsamples} samples along line {p0} -> {p1}...", end=" ", flush=True)

	# pick one
	samples = sample_opencv(im, M, nsamples) # does "normal" cubic (4 support points, continuous first derivative)
	#samples = sample_scipy(im, M, nsamples) # does some fancy cubic with continuous higher derivatives

	print("sampling done")

	# smoothing to remove noise
	if smoothing_sigma > 0:
		samples = scipy.ndimage.gaussian_filter1d(samples, sigma=smoothing_sigma / stride)

	# off-by-half in position because for values [0,1,1,0] this returns [+1,0,-1]
	gradient = np.diff(samples) / stride

	i_falling = np.argmin(gradient) # in samples
	i_rising = np.argmax(gradient) # in samples

	distance = (i_rising - i_falling) * stride # in pixels

	print(f"distance: {distance:.{decimals}f} pixels")

	# this was the result. algorithm is done.
	# now follows displaying code

	if do_display:
		gradient *= 255 / np.abs(gradient).max()

		# plot signal
		plot = cv.plot.Plot2d_create(np.arange(nsamples, dtype=np.float64), samples.astype(np.float64))
		plot.setMinY(256+32)
		plot.setMaxY(-32)
		plot.setMinX(0)
		plot.setMaxX(nsamples)
		plot.setGridLinesNumber(5)
		plot.setShowText(False) # callout for specific point, setPointIdxToPrint(index)
		plot.setPlotGridColor((64,)*3)
		canvas1 = plot.render()

		# plot gradient
		plot = cv.plot.Plot2d_create(np.arange(nsamples-1) + 0.5, gradient.astype(np.float64))
		plot.setMinY(256+64)
		plot.setMaxY(-256-64)
		plot.setMinX(0)
		plot.setMaxX(nsamples)
		plot.setGridLinesNumber(5)
		plot.setShowText(False) # callout for specific point, setPointIdxToPrint(index)
		plot.setPlotGridColor((64,)*3)
		canvas2 = plot.render()

		# arrange vertically
		canvas = np.vstack([canvas1, canvas2]) # 600 wide, 800 tall

		# draw lines at edges (largest gradients)
		# plots are 600x400 pixels... and there's no way to plot multiple or plot lines in "plot space"
		px_falling = int(600 * (i_falling+0.5) / nsamples)
		px_rising = int(600 * (i_rising+0.5) / nsamples)
		cv.line(canvas, (px_falling, 0), (px_falling, 400*2), color=(255,0,0))
		cv.line(canvas, (px_rising, 0), (px_rising, 400*2), color=(255,0,0))

		# some text to describe the picture
		cv.putText(canvas, f"sampling {p0} -> {p1}",
			(10, 350), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), thickness=1, lineType=cv.LINE_AA)

		cv.putText(canvas, f"stride {stride} px, {nsamples} samples, sigma {smoothing_sigma}",
			(10, 350+35), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), thickness=1, lineType=cv.LINE_AA)

		cv.putText(canvas, f"distance: {distance:.{decimals}f} px",
			(10, 350+70), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), thickness=1, lineType=cv.LINE_AA)

		# save for posterity
		cv.imwrite("plot.png", canvas)

		cv.imshow("plot", canvas)

		print("press Ctrl+C in the terminal, or press any key while the imshow() window is focused")

		while True:
			keycode = cv.waitKey(100)
			if keycode == -1:
				continue
			else:
				print(f"keycode: {keycode}")
				break

