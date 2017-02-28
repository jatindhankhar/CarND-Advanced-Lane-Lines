import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import helper
from calibrate import calibrate
from moviepy.editor import VideoFileClip
from lane import Lanes
import os.path


def draw_lane_line(warped, undist, lanes):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array(
        [np.transpose(np.vstack([lanes.left_lane.best_fit, lanes.ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([lanes.right_lane.best_fit, lanes.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective
    # matrix (Minv)
    newwarp = helper.warp_transform(color_warp, reverse=True)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Calculate curvatures and offset from center of the lane
    curvatures = lanes.curvature()
    curvatures_text = "left line radius: {:8.2f}m, right line radius: {:8.2f}m".format(
        float(curvatures[0]), float(curvatures[1]))
    cv2.putText(result, curvatures_text, (10, 20),
                cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255))
    offset_from_center = (result.shape[
                          1] / 2 - (lanes.left_lane.best_fit[-1] + lanes.right_lane.best_fit[-1]) / 2) * 3.7 / 700
    offset_from_center_text = "offset from center of the line: {:2.5f}m".format(
        offset_from_center)
    cv2.putText(result, offset_from_center_text, (10, 40),
                cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255))
    return result


def pipeline(image, lanes):
    undistorted = helper.undistort(image, lanes.mtx, lanes.dist)
    # find lane line by threshold function
    thresholded = helper.combined_threshold(undistorted)
    hls_out = helper.hls_select(undistorted)
    warped = helper.warp_transform(thresholded)  # get warped binary image
    color_warped = helper.warp_transform(undistorted)
    hls_warped = helper.hls_select(color_warped)
    color_reverse_warped = helper.warp_transform(undistorted, reverse=True)
    lanes.detect_lanes(warped)  # find lane lines and draw area
    processed = draw_lane_line(warped, undistorted, lanes)

    # Wonderful Pipeline by John Chen and Yu Shen
    # https://carnd-forums.udacity.com/questions/32706990/want-to-create-a-diagnostic-view-into-your-lane-finding-pipeline
    font = cv2.FONT_HERSHEY_COMPLEX
    middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
    lanes.curvatures()
    cv2.putText(middlepanel, "Average lane curvature: {:8.2f} m".format(
        np.average(lanes.curvature())), (30, 60), font, 1, (0, 255, 0), 2)
    offset_from_center = (result.shape[
                          1] / 2 - (lanes.left_lane.best_fit[-1] + lanes.right_lane.best_fit[-1]) / 2) * 3.7 / 700
    offset_from_center_text = "offset from center of the line: {:2.5f}m".format(
        offset_from_center)
    cv2.putText(middlepanel, offset_from_center_text,
                (30, 90), font, 1, (0, 255, 0), 2)
    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    diagScreen[0:720, 0:1280] = processed
    diagScreen[0:240, 1280:1600] = cv2.resize(
        undistorted, (320, 240), interpolation=cv2.INTER_AREA)
    diagScreen[0:240, 1600:1920] = cv2.resize(helper.to_RGB(
        thresholded), (320, 240), interpolation=cv2.INTER_AREA)
    diagScreen[240:480, 1280:1600] = cv2.resize(
        helper.to_RGB(warped), (320, 240), interpolation=cv2.INTER_AREA)
    diagScreen[240:480, 1600:1920] = cv2.resize(helper.to_RGB(
        hls_out), (320, 240), interpolation=cv2.INTER_AREA) * 4
    #diagScreen[600:1080, 1280:1920] = cv2.resize(diag7, (640,480), interpolation=cv2.INTER_AREA)*4
    diagScreen[720:840, 0:1280] = middlepanel
    diagScreen[840:1080, 0:320] = cv2.resize(helper.to_RGB(
        hls_warped), (320, 240), interpolation=cv2.INTER_AREA)
    diagScreen[840:1080, 320:640] = cv2.resize(
        color_warped, (320, 240), interpolation=cv2.INTER_AREA)
    diagScreen[840:1080, 640:960] = cv2.resize(
        color_reverse_warped, (320, 240), interpolation=cv2.INTER_AREA)
    #diagScreen[840:1080, 960:1280] = cv2.resize(diag8, (320,240), interpolation=cv2.INTER_AREA)
    return diagScreen


def process_video(single_frame=False):
    #_mtx,_dist = helper.load_data()
    print("Processing Video ...")
    destination = 'processed.mp4'
    source = VideoFileClip("project_video.mp4")
    lane = Lanes()
    lane.mtx, lane.dist = helper.load_data()
    video = source.fl_image(lambda frame: pipeline(frame, lane))
    video.write_videofile(destination, audio=False)
    print("Processing Done ...")

if __name__ == '__main__':
    # If Pickle doesn't exist create one, calibrate first then proces video
    if not (os.path.exists("wide_dist_pickle.p") and os.path.isfile("wide_dist_pickle.p")):
        calibrate(verbose=True)

    process_video()
