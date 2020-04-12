import cv2
import numpy as np


FILENAME = "frame_capture_1.mp4"

cap = cv2.VideoCapture(FILENAME)
start_frame_number = 5
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)


while True:	
	# Now when you read the frame, you will be reading the 50th frame
	success, frame = cap.read()
	cv2.imwrite('frame' + str(start_frame_number) + '.png',frame)
	start_frame_number += 200
	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
		

cap.release()