import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

# termination criteria for corner detection in the chessboard
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points of the cheassboard for camera calibration 
nx = 9#TODO: enter the number of inside corners in x
ny = 6#TODO: enter the number of inside corners in y

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3),np.float32)
objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./camera_cal/*.jpg')

print('--> Hint: all informations and images of the different processing steps can be found in the ''outputs'' folder')
print('')
print('* Step1: CameraCalibration /Determine cam intrinsics')
print('')
print('[CameraCalibration]...Started')

for idx,fname in enumerate(images):
	# Make a list of calibration images
	img = cv2.imread(fname)
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

	# If found, draw corners
	if ret == True:
    # Draw and display the corners
		#print('Debug:')
		print('--> working on: ' ,fname)
		objpoints.append(objp)
		corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
		imgpoints.append(corners2)
		cv2.drawChessboardCorners(img, (nx, ny), corners2, ret)
		write_name='Step1_CameraCalibration_AllCornersDetected'+str(idx)+'.jpg'
		cv2.imwrite('./output_images/'+write_name,img)
		plt.imshow(img)
		plt.show
	
#preparation of calc the camera distortion	
img = cv2.imread('./camera_cal/calibration2.jpg')
img_size = (img.shape[1],img.shape[0])	

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,img_size,None,None)
	
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

reproError =mean_error/len(objpoints)
    
print("--> ReprojectionError[pixel]: {} ".format(reproError) )    

#validation of the intrinsic calibration data via reprojection error
if (reproError>1.0):
    print('--> Validation: Calibration is too bad. You have to repeate taking pictures!')
if ((reproError<1.0) and (reproError>0.1)):
    print('--> Validation: Calibration is good for use!')
if (reproError<0.1):
    print('--> Validation: Calibration is perfect')     
print('[CameraCalibration]...Done')   
    
# save the camera calibration coefficients for undistorting images later
dist_pickle={}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
dist_pickle["rvecs"]= rvecs
dist_pickle["tvecs"]= tvecs
dist_pickle["repro_error"]= reproError

dist_pickle = pickle.dump(dist_pickle, open( "./output_images/CameraCalibration_pickle.p", "wb" ) )
print('[CameraCalibration]...Write calibration data to file: CameraCalibration_pickle.p')
