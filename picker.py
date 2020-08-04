#!/usr/bin/env python3

# written in 2020 by Christoph Rackwitz <christoph.rackwitz@gmail.com>
# use only for war purposes and sarcasm

import sys
import time
import numpy as np
import cv2 as cv

if __name__ == '__main__':
	windowname = "coordinate picker"

	print("move your mouse, click and drag")

	im = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
	imh,imw = im.shape[:2]
	im = cv.cvtColor(im, cv.COLOR_GRAY2BGR) # need it gray because I'm gonna draw on it in color

	p0 = p1 = None

	invalid = True

	def invalidate():
		global invalid
		invalid = True

	def redraw():
		global invalid
		if p0 and p1:
			canvas = im.copy()
			cv.line(canvas, p0, p1, color=(0,0,255), thickness=2, lineType=cv.LINE_AA)
		else:
			canvas = im
		cv.imshow(windowname, canvas)
		invalid = False
		#print("redrawn at", time.perf_counter())

	def onmouse(event, x, y, flags, userdata):
		global p0, p1

		print(f"{(x,y)}", end="\r", flush=True)

		if flags == cv.EVENT_LBUTTONDOWN:
			p1 = (x,y)
			#print("p1 =", p1, end="\r", flush=True)
			invalidate()

		if event == cv.EVENT_LBUTTONDOWN:
			p0 = (x,y)
			#print("p0 =", p0)
			invalidate()

		if event == cv.EVENT_LBUTTONUP:
			p1 = (x,y)
			invalidate()

			x0,y0 = p0
			x1,y1 = p1
			print(f"{p0} -> {p1}  --line={x0},{y0},{x1},{y1}")

	# NORMAL for resizability
	# OPENGL because it has linear resampling at least (default on windows is nearest neighbor...)
	cv.namedWindow(windowname, cv.WINDOW_NORMAL | cv.WINDOW_OPENGL)

	# some nice default size
	scale = np.sqrt(1e6 / (imw*imh))
	cv.resizeWindow(windowname, int(imw * scale), int(imh * scale))

	cv.setMouseCallback(windowname, onmouse)

	while True:
		if invalid:
			redraw()

		key = cv.waitKey(10) # not too fast, not too slow
		if key == -1: continue

		if key in (13, 27):
			break


	cv.destroyAllWindows()
	print("\ndone")

