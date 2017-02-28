import numpy as np
import cv2


class Lane():

    def __init__(self):
        self.best_fit = None
        self.current_fit = None


class Lanes():

    def __init__(self):
        self.detected = False
        self.ploty = None
        self.mtx = None
        self.dist = None
        self.left_lane = Lane()
        self.right_lane = Lane()

    def curvature(self):
        # Code based upon https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/2f928913-21f6-4611-9055-01744acc344f
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(
            self.ploty * ym_per_pix, self.left_lane.best_fit * xm_per_pix, 2)
        right_fit_cr = np.polyfit(
            self.ploty * ym_per_pix, self.right_lane.best_fit * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * np.max(
            self.ploty) * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * np.max(
            self.ploty) * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
    # print(left_curverad,'m',right_curverad,'m')
        # Now our radius of curvature is in meters
        return (left_curverad, right_curverad)

    def detect_lanes(self, binary_warped):
        if self.detected:
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            margin = 50
            left_lane_inds = ((nonzerox > (self.left_lane.current_fit[0] * (nonzeroy**2) + self.left_lane.current_fit[1] * nonzeroy + self.left_lane.current_fit[2] - margin)) &
                              (nonzerox < (self.left_lane.current_fit[0] * (nonzeroy**2) + self.left_lane.current_fit[1] * nonzeroy + self.left_lane.current_fit[2] + margin)))
            right_lane_inds = ((nonzerox > (self.right_lane.current_fit[0] * (nonzeroy**2) + self.right_lane.current_fit[1] * nonzeroy + self.right_lane.current_fit[2] - margin)) &
                               (nonzerox < (self.right_lane.current_fit[0] * (nonzeroy**2) + self.right_lane.current_fit[1] * nonzeroy + self.right_lane.current_fit[2] + margin)))

            # if (left_lane_inds.size == 0 and right_fit.inds.size == 0):
            #   self.detected = False
#    self.detect_lanes()
            # return

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            self.left_lane.current_fit = np.polyfit(lefty, leftx, 2)
            self.right_lane.current_fit = np.polyfit(righty, rightx, 2)
            self.ploty = np.linspace(0, binary_warped.shape[
                                     0] - 1, binary_warped.shape[0])
            self.left_lane.best_fit = self.left_lane.current_fit[
                0] * self.ploty**2 + self.left_lane.current_fit[1] * self.ploty + self.left_lane.current_fit[2]
            self.right_lane.best_fit = self.right_lane.current_fit[
                0] * self.ploty**2 + self.right_lane.current_fit[1] * self.ploty + self.right_lane.current_fit[2]

        else:
            # Assuming you have created a warped binary image called "binary_warped"
            # Take a histogram of the bottom half of the image
            histogram = np.sum(
                binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack(
                (binary_warped, binary_warped, binary_warped)) * 255
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            window_height = np.int(binary_warped.shape[0] / nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Set the width of the windows +/- margin
            margin = 100
            # Set minimum number of pixels found to recenter window
            minpix = 50
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                    # Identify window boundaries in x and y (and right and
                    # left)
                win_y_low = binary_warped.shape[
                    0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 2)
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                    nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                    nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their
                # mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Fit a second order polynomial to each
            self.left_lane.current_fit = np.polyfit(lefty, leftx, 2)

            # Generate x and y values for plotting
            self.right_lane.current_fit = np.polyfit(righty, rightx, 2)
            self.ploty = np.linspace(0, binary_warped.shape[
                                     0] - 1, binary_warped.shape[0])
            self.left_lane.best_fit = self.left_lane.current_fit[
                0] * self.ploty**2 + self.left_lane.current_fit[1] * self.ploty + self.left_lane.current_fit[2]
            self.right_lane.best_fit = self.right_lane.current_fit[
                0] * self.ploty**2 + self.right_lane.current_fit[1] * self.ploty + self.right_lane.current_fit[2]
            self.detected = True

        return
