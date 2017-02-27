import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import pickle

def undistort(img,mtx,dist):
    """ Undistorts an image """
    return cv2.undistort(img, mtx, dist, None, mtx)



def load_data():
    """Loads the pickle data and returns a tuple"""
    print("Loading the data ...")
    file = open("wide_dist_pickle.p",'rb')
    object_file = pickle.load(file)
    file.close()
    return (object_file['mtx'],object_file['dist'])

def pickle_data(objpoints,imgpoints):
    """ Pickles the data """
    print("Saving the data ...")
    img = cv2.imread('camera_cal/calibration11.jpg')
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "wide_dist_pickle.p", "wb" ) )

def calibrate(verbose=False):
    """ Calibrate the camera for further processing """
    objp = np.zeros((6*9,3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    objpoints = []
    imgpoints = []
    failed_images = []
    nx = 9 # No of horizontal centers in the chessboard
    ny = 6 # No of vertical centers in the chessboard
    images = glob.glob('camera_cal/calibration*.jpg')
    print("Calibrating ....")
    for image in images:
        img = mpimg.imread(image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
        ret,corners = cv2.findChessboardCorners(gray,(nx,ny),None)
    # If found, add object points and image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw the chessboardCorners
            cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
            #target_name = "output_images/"  + os.path.basename(image).replace("calibration","calibrated")
            if verbose:
                print("Worked for : " + image)
                plt.figure()
                plt.imshow(img)
                #cv2.imwrite(target_image,img)
                #cv2.imshow('img',img)
                #cv2.waitKey(500)
            else:
                if verbose:
                    print("Unable to find Points for " + image)
                failed_images.append(image)
    print("Calibration done ....")
    pickle_data(objpoints,imgpoints)



if __name__ == "__main__":
    calibrate(verbose=True)
