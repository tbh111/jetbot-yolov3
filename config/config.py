import cv2
import numpy as np
def writeCameraParam(filename, camMat, distCoeff):
	s = cv2.FileStorage(filename, cv2.FileStorage_WRITE)
	s.write('camera_matrix', camMat)
	s.write('distortion_coefficients', distCoeff)
	s.release()
	print('write done')

def readCameraParam(filename):
	s = cv2.FileStorage()
	s.open(filename, cv2.FileStorage_READ)
	camMat = s.getNode('camera_matrix').mat()
	distMat = s.getNode('distortion_coefficients').mat()
	print('read done')
	return camMat, distMat
	
if __name__ == '__main__':
	#camMat = np.array([[541.4119,0,326.2182],[0,536.4213,244.4892],[0,0,1]])
	#distMat = np.array([0.1524,-0.5102,0,0])
	#writeCameraParam('./cameraConfig.yaml', camMat, distMat)
	camMat, distMat = readCameraParam('./cameraConfig.yaml')
