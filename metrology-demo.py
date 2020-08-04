#!/usr/bin/env python3

# written in 2020 by Christoph Rackwitz <christoph.rackwitz@gmail.com>
# use only for war purposes and sarcasm

import sys
import argparse
import numpy as np
import cv2 as cv

import scipy.ndimage
# contains lots of useful stuff that's also in OpenCV
# https://scipy.github.io/devdocs/ndimage.html


### "business logic" ###################################################

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
	# available: INTER_{NEAREST,LINEAR,AREA,CUBIC,LANCOS4)
	samples = cv.warpAffine(im, M=M, dsize=(nsamples, 1), flags=cv.WARP_INVERSE_MAP | cv.INTER_CUBIC )

	# flatten row vector
	samples.shape = (-1,)

	# INTER_CUBIC seems to break down beyond 1/32 sampling (discretizes).
	# there might be fixed point algorithms at work

	return samples

def sample_scipy(im, M, nsamples):
	# coordinates to this function are (i,j) = (y,x)
	# I could permute first and second rows+columns of M, or transpose input+output
	Mp = M.copy()
	Mp[(0,1), :] = Mp[(1,0), :] # permute rows
	Mp[:, (0,1)] = Mp[:, (1,0)] # permute columns

	samples = scipy.ndimage.interpolation.affine_transform(
		input=im, matrix=Mp, output_shape=(1, nsamples),
		order=2, # 1: linear (C0, f' is piecewise constant), 2: C1 (f' is piecewise linear), 3: C2... https://en.wikipedia.org/wiki/Smoothness
		mode='nearest' # border handling
	)

	# flatten row vector
	samples.shape = (-1,)

	return samples

### command line parsing utility functions #############################

def parse_linestr(arg):
	pieces = arg.split(",")
	pieces = [float(el) for el in pieces]
	x0,y0,x1,y1 = pieces
	return ((x0,y0), (x1,y1))

def parse_bool(arg):
	if isinstance(arg, bool):
	   return arg
	if arg.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError(f'Boolean value expected, got {arg!r} instead')

def parse_float(arg):
	import ast

	if '/' in arg:
		num, denom = arg.split('/', 1)
		num = ast.literal_eval(num)
		denom = ast.literal_eval(denom)
		result = num / denom

	else:
		result = ast.literal_eval(arg)

	return result

### main... ############################################################

if __name__ == '__main__':
	# command line argument parsing
	# change defaults here

	parser = argparse.ArgumentParser()
	parser.add_argument("--picture", dest="fname", metavar="PATH", type=str, default="dish-1.jpg",
		help="path to picture file")
	parser.add_argument("--invert", type=parse_bool, default=True, metavar="BOOL",
		help="invert picture (cosmetic; distance between gradient extrema is absolute)")
	parser.add_argument("--line", type=parse_linestr, default=((1320, 2500), (1320, 2100)), metavar="X0,Y0,X1,Y1",
		help="line to sample on")
	parser.add_argument("--stride", type=parse_float, default=1/4, metavar="PX",
		help="stride in pixels to sample along line, fractions supported")
	parser.add_argument("--method", type=lambda s: s.lower(), default="scipy",
		help="sampling methods: SciPy (slower, smoother, default), OpenCV (faster, less smooth)")
	parser.add_argument("--sigma", type=float, default=2.0, metavar="PX",
		help="sigma for gaussian lowpass on sampled signal, before gradient is calculated")
	parser.add_argument("--verbose", type=parse_bool, default=True, metavar="BOOL",
		help="chatty or not")
	parser.add_argument("--display", type=parse_bool, default=True, metavar="BOOL",
		help="draw some plots")
	parser.add_argument("--saveplot", type=str, default="plot.png", metavar="PATH",
		help="save a picture (use '--saveplot=' to disable)")
	args = parser.parse_args()

	########## here be dragons ##########

	if args.stride > 1:
		print(f"WARNING: stride should be <= 1, is {args.stride}")

	stride_decimals = max(0, int(np.ceil(-np.log10(args.stride))))

	if args.verbose: print("loading picture...", end=" ", flush=True)
	im = cv.imread(args.fname, cv.IMREAD_GRAYSCALE)
	imh, imw = im.shape[:2]
	if args.invert:
		im = 255-im # invert
	im = im.astype(np.float32)# * np.float32(1/255)
	if args.verbose: print("done")

	# build transform
	p0, p1 = args.line
	nsamples, M = build_transform(p0, p1, stride=args.stride)

	if args.verbose: print(f"taking {nsamples} samples along line {p0} -> {p1}...", end=" ", flush=True)

	# pick one
	if args.method == 'opencv':
		samples = sample_opencv(im, M, nsamples) # does "normal" cubic (4x4 support points, continuous first derivative)
	elif args.method == 'scipy':
		samples = sample_scipy(im, M, nsamples) # does some fancy "cubic" with continuous higher derivatives
	else:
		assert False, "method needs to be opencv or scipy"

	if args.verbose: print("sampling done")

	# smoothing to remove noise
	if args.sigma > 0:
		if args.verbose: print(f"lowpass filtering with sigma = {args.sigma} px...", end=" ", flush=True)
		samples = scipy.ndimage.gaussian_filter1d(samples, sigma=args.sigma / args.stride)
		if args.verbose: print("done")

	# off-by-half in position because for values [0,1,1,0] this returns [+1,0,-1]
	gradient = np.diff(samples) / args.stride

	i_falling = np.argmin(gradient) # in samples
	i_rising = np.argmax(gradient) # in samples

	distance = np.abs(i_rising - i_falling) * args.stride # in pixels

	if args.verbose:
		print(f"distance: {distance:.{stride_decimals}f} pixels")
	else:
		print(distance)

	# this was the result. algorithm is done.
	# now follows displaying code

	if args.display or args.saveplot:
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
		cv.putText(canvas, f"{nsamples*args.stride:.0f} px, {p0} -> {p1}",
			(10, 350), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), thickness=1, lineType=cv.LINE_AA)

		cv.putText(canvas, f"stride {args.stride} px, {nsamples} samples, sigma {args.sigma}",
			(10, 350+35), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), thickness=1, lineType=cv.LINE_AA)

		cv.putText(canvas, f"distance: {distance:.{stride_decimals}f} px",
			(10, 350+70), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), thickness=1, lineType=cv.LINE_AA)

		# save for posterity
		if args.saveplot:
			cv.imwrite(args.saveplot, canvas)

		if args.display:
			cv.imshow("plot", canvas)

			if args.verbose:
				print("press Ctrl+C in the terminal, or press any key while the imshow() window is focused")

			while True:
				keycode = cv.waitKey(100)

				if keycode == -1:
					continue

				# some key...

				if args.verbose:
					print(f"keycode: {keycode}")

				cv.destroyAllWindows()
				break

