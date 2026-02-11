import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print("✅ Frame type:", type(frame), "Shape:", frame.shape if ret else "No frame")
cap.release()
