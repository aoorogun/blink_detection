from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import random
import time
import dlib
import cv2

def both_eyes():

	EYE_AR_THRESH = 0.2
	EYE_AR_CONSEC_FRAMES = 4 #Number of consecutive frames
	COUNTER = 0
	TOTAL = 0

	# start the video stream thread
	print("[INFO] starting video stream thread...")

	vs = VideoStream(src=0).start()
	fileStream = False
	time.sleep(1.0)

	# loop over frames from the video stream
	while True:

		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detect faces in the grayscale frame
		rects = detector(gray, 0)

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0
			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes

			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)

			if args["show_contour"] > 0:
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1
			# otherwise, the eye aspect ratio is not below the blink
			# threshold
			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1
				# reset the eye frame counter
				COUNTER = 0

			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	
		# show the frame
		cv2.imshow("Blink both Eyes", frame)
		key = cv2.waitKey(1) & 0xFF
	
		# press q to break from the loop
		if key == ord("q"):
			break
	#clean up
	cv2.destroyAllWindows()
	vs.stop()

def check_right():

	EYE_AR_THRESH = 0.13
	EYE_AR_CONSEC_FRAMES = 4 #Number of consecutive frames
	COUNTER = 0
	TOTAL = 0

	# Check Right EYE
	print("[INFO] starting video stream thread...")
	# vs = FileVideoStream(args["video"]).start()
	# fileStream = True
	vs = VideoStream(src=0).start()
	fileStream = False
	time.sleep(1.0)

	# loop over frames from the video stream
	while True:
		# if this is a file video stream, then we need to check if
		# there any more frames left in the buffer to process
		if fileStream and not vs.more():
			break
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detect faces in the grayscale frame
		rects = detector(gray, 0)

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			# extract the right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for right eye
			#leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			#leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			# average the eye aspect ratio together for both eyes
			ear = rightEAR / 2.0
			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes

			# leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)

			if args["show_contour"] > 0:
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1
			# otherwise, the eye aspect ratio is not below the blink
			# threshold
			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1
				# reset the eye frame counter
				COUNTER = 0

			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	
		# show the frame
		cv2.imshow("Blink Right Eye", frame)
		key = cv2.waitKey(1) & 0xFF
	
		# press q to break from the loop
		if key == ord("q"):
			break
	#clean up
	cv2.destroyAllWindows()
	vs.stop()

def check_left():

	EYE_AR_THRESH = 0.13
	EYE_AR_CONSEC_FRAMES = 4 #Number of consecutive frames
	COUNTER = 0
	TOTAL = 0


	# Check Right EYE
	print("[INFO] starting video stream thread...")
	# vs = FileVideoStream(args["video"]).start()
	# fileStream = True
	vs = VideoStream(src=0).start()
	fileStream = False
	time.sleep(1.0)

	# loop over frames from the video stream
	while True:
		# if this is a file video stream, then we need to check if
		# there any more frames left in the buffer to process
		if fileStream and not vs.more():
			break
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detect faces in the grayscale frame
		rects = detector(gray, 0)

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			# extract the right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for right eye
			leftEye = shape[lStart:lEnd]
			#rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			#rightEAR = eye_aspect_ratio(rightEye)
			# average the eye aspect ratio together for both eyes
			ear = leftEAR / 2.0
			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes

			leftEyeHull = cv2.convexHull(leftEye)
			# rightEyeHull = cv2.convexHull(rightEye)
			if args["show_contour"] > 0:
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			#cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1
			# otherwise, the eye aspect ratio is not below the blink
			# threshold
			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1
				# reset the eye frame counter
				COUNTER = 0

			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	
		# show the frame
		cv2.imshow("Blink left eye", frame)
		key = cv2.waitKey(1) & 0xFF
	
		# press q to break from the loop
		if key == ord("q"):
			break
	#clean up
	cv2.destroyAllWindows()
	vs.stop()

if __name__ == "__main__":
	LANDMARK_MODEL = ""
	ap = argparse.ArgumentParser()
	ap.add_argument("-lm", "--landmark-model", required=True, help="path to facial landmark predictor")
	ap.add_argument("-sc", "--show-contour", type=int, default=0, help="visualize eye landmark contour. 0 to not view, 1 to view")

	args = vars(ap.parse_args())

	print("Loading Model")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["landmark_model"])

	
	# define eye aspect ratio http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

	def eye_aspect_ratio(eye):
		# compute the euclidean distances between the two sets of
		# vertical eye landmarks (x, y)-coordinates
		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])
		# compute the euclidean distance between the horizontal
		# eye landmark (x, y)-coordinates
		C = dist.euclidean(eye[0], eye[3])
		ear = (A + B) / (2.0 * C)

		return ear

	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	sample = [both_eyes, check_right, check_left]
	random_biom = random.sample(sample, 2)

	#shuffle order
	random.shuffle(random_biom)

	for biom in random_biom:
		biom()