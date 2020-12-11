
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lane_line import *


def fit_poly(img_shape, line_x, line_y):
    # Fit a second order polynomial to each with np.polyfit()
    poly_fit = np.polyfit(line_y, line_x, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit
    fit_x = poly_fit[0] * ploty**2 + poly_fit[1] * ploty + poly_fit[2]

    return fit_x, ploty, poly_fit


# HYPERPARAMETERS
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50


def find_lane_pixels(binary_warped, left_line, right_line):

    def get_histogram(binary_img, y_low, y_high=None):
        if y_high == None:
            return np.sum(binary_img[y_low:, :], axis=0)
        else:
            return np.sum(binary_img[y_low:y_high, :], axis=0)

    def get_window_x_center(line, ploty):
        poly_fit = line.current_fit
        fit_x = poly_fit[0] * ploty**2 + poly_fit[1] * ploty + poly_fit[2]
        x_center = np.int(np.mean(fit_x))
        return x_center

    # Take a histogram of the bottom half of the image
    # histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:,:], axis=0)
    histogram = get_histogram(binary_warped, binary_warped.shape[0] // 2)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    left_x_center = np.argmax(
        histogram[:midpoint]) if left_line.detected == False else None
    right_x_center = (np.argmax(
        histogram[midpoint:]) + midpoint) if right_line.detected == False else None

    # Create empty lists to receive left and right lane pixel indices
    # left_lane_inds = []
    # right_lane_inds = []
    l_windows = []
    r_windows = []
    # empty_threshold = margin * window_height // 10
    empty_threshold = 10
    continuous_threshold = nwindows // 2 - 1
    l_continuous_empty = 0
    r_continuous_empty = 0

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        # Compute left_x_center and right_x_center
        ploty = np.linspace(win_y_low, win_y_high, window_height)
        if left_line.detected == True:
            if l_continuous_empty >= continuous_threshold:
                histogram = get_histogram(binary_warped, 0, win_y_high)
                max_value = np.max(histogram[:midpoint])
                if max_value > 0:
                    left_x_center = np.argmax(histogram[:midpoint])
            else:
                left_x_center = get_window_x_center(left_line, ploty)

        if right_line.detected == True:
            if r_continuous_empty >= continuous_threshold:
                histogram = get_histogram(binary_warped, 0, win_y_high)
                max_value = np.max(histogram[midpoint:])
                if max_value > 0:
                    right_x_center = np.argmax(histogram[midpoint:]) + midpoint
            else:
                right_x_center = get_window_x_center(right_line, ploty)
        # Find the four below boundaries of the window
        l_win = Window()
        r_win = Window()
        # Set left/right window boundary
        l_win.y_low = win_y_low
        l_win.y_high = win_y_high
        l_win.x_low = left_x_center - margin
        l_win.x_high = left_x_center + margin
        r_win.y_low = win_y_low
        r_win.y_high = win_y_high
        r_win.x_low = right_x_center - margin
        r_win.x_high = right_x_center + margin

        # Identify the nonzero pixels in x and y within the window
        left_good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= l_win.x_low) & (nonzerox < l_win.x_high)).nonzero()[0]
        right_good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= r_win.x_low) & (nonzerox < r_win.x_high)).nonzero()[0]

        # Monitor the number of continuous empty window
        # TODO: I want to do some verification here
        if left_good_inds.shape[0] <= empty_threshold:
            l_continuous_empty += 1
        else:
            l_continuous_empty = 0
        if right_good_inds.shape[0] <= empty_threshold:
            r_continuous_empty += 1
        else:
            r_continuous_empty = 0

        left_x = nonzerox[left_good_inds]
        left_y = nonzeroy[left_good_inds]
        right_x = nonzerox[right_good_inds]
        right_y = nonzeroy[right_good_inds]
        l_win.allx = left_x
        l_win.ally = left_y
        r_win.allx = right_x
        r_win.ally = right_y

        # If you found > minpix pixels, recenter next window
        # (`right` or `leftx_current`) on their mean position
        if len(left_good_inds) > minpix and left_line.detected == False:
            left_x_center = np.int(np.mean(nonzerox[left_good_inds]))
        if len(right_good_inds) > minpix and right_line.detected == False:
            right_x_center = np.int(np.mean(nonzerox[right_good_inds]))

        l_windows.append(l_win)
        r_windows.append(r_win)
    return l_windows, r_windows


# Define conversions in x and y from pixels space to meters
ym_per_pix = 3 / 180  # meters per pixel in y dimension
xm_per_pix = 3.7 / 610  # meters per pixel in x dimension


def measure_curvature_real(plotx, ploty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    # ym_per_pix = 30 / 720 # meters per pixel in y dimension
    # xm_per_pix = 3.7 / 700 # meters per pixel in x dimension

    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    fit_cr = np.polyfit(ploty * ym_per_pix, plotx * xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    curverad = (1 + (2 * fit_cr[0] * y_eval * ym_per_pix
                     + fit_cr[1])**2)**1.5 / np.absolute(2 * fit_cr[0])

    return curverad


def draw_lane_pixel(binary_warped, left_line, right_line, ploty):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    out_img[left_line.ally, left_line.allx] = [255, 0, 0]
    out_img[right_line.ally, right_line.allx] = [0, 0, 255]

    # Draw best fit
    l_best_fitx = left_line.best_fit[0] * ploty**2 + \
        left_line.best_fit[1] * ploty + left_line.best_fit[2]
    r_best_fitx = right_line.best_fit[0] * ploty**2 + \
        right_line.best_fit[1] * ploty + right_line.best_fit[2]
    for x1, x2, y in zip(l_best_fitx, r_best_fitx, ploty):
        # yellow
        cv2.circle(out_img, (int(x1), int(y)), 2, (252, 246, 66), -1)
        cv2.circle(out_img, (int(x2), int(y)), 2, (252, 246, 66), -1)
    # Draw current fit
    l_curr_fitx = left_line.current_fit[0] * ploty**2 + \
        left_line.current_fit[1] * ploty + left_line.current_fit[2]
    r_curr_fitx = right_line.current_fit[0] * ploty**2 + \
        right_line.current_fit[1] * ploty + right_line.current_fit[2]
    for x1, x2, y in zip(l_curr_fitx, r_curr_fitx, ploty):
        # purple
        cv2.circle(out_img, (int(x1), int(y)), 2, (217, 48, 214), -1)
        cv2.circle(out_img, (int(x2), int(y)), 2, (217, 48, 214), -1)
    # Draw windows
    for win in left_line.windows:
        info = '{:>5d}'.format(win.allx.shape[0])
        cv2.putText(out_img, info, (win.x_high + 1, (win.y_low + win.y_high) //
                                    2 + 15), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.rectangle(out_img, (win.x_low, win.y_high),
                      (win.x_high, win.y_low), (0, 255, 0), 2)
    for win in right_line.windows:
        info = '{:>5d}'.format(win.allx.shape[0])
        cv2.putText(out_img, info, (win.x_high + 1, (win.y_low + win.y_high) //
                                    2 + 15), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.rectangle(out_img, (win.x_low, win.y_high),
                      (win.x_high, win.y_low), (0, 255, 0), 2)
    cv2.rectangle(
        out_img, (0, 0), (binary_warped.shape[1], binary_warped.shape[0]), (0, 255, 0), 5)
    return out_img


def update_line(left_line, leftx, lefty, right_line, rightx, righty):
    left_poly = np.polyfit(lefty, leftx, 2)
    left_line.update_curr_fit(left_poly)
    left_line.allx, left_line.ally = leftx, lefty
    left_line.radius_of_curvature = measure_curvature_real(leftx, lefty)


def sanity_checking(image, left_line, right_line):
    def get_fit_x(poly_fit, ploty):
        return poly_fit[0] * ploty**2 + poly_fit[1] * ploty + [2]

    def check_range(image, x):
        return x >= 0 and x < image.shape[1]

    def check_outlier(data):
        outlier_constant = 1.5
        upper_quartile = np.percentile(data, 75)
        lower_quartile = np.percentile(data, 25)
        iqr = (upper_quartile - lower_quartile) * outlier_constant
        upper_bound = upper_quartile + iqr
        lower_bound = lower_quartile - iqr
        outlier = data[(data < lower_bound) | (data > upper_bound)]
        return outlier.shape[0] > 0

    if left_line.detected == False or right_line.detected == False:
        return False

    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = get_fit_x(left_line.best_fit, ploty)
    right_fitx = get_fit_x(right_line.best_fit, ploty)

    check_point = 10
    check_ploty = np.linspace(0, image.shape[0] - 1, check_point)
    check_left_fitx = get_fit_x(left_line.best_fit, check_ploty)
    check_right_fitx = get_fit_x(right_line.best_fit, check_ploty)
    x_distance = check_right_fitx - check_left_fitx
    min_dis = np.min(x_distance)
    max_dis = np.max(x_distance)

    left_radius = left_line.measure_curvature_real(
        left_fitx, ploty, y_eval=check_ploty)
    right_radius = right_line.measure_curvature_real(
        right_fitx, ploty, y_eval=check_ploty)

    y_top_idx = None
    y_bottom_idx = None
    result = True
    for idx in range(len(check_ploty)):
        if check_range(image, check_left_fitx[idx]) and check_range(image, check_right_fitx[idx]):
            y_top_idx = idx
            break
    for idx in range(len(check_ploty) - 1, -1, -1):
        if check_range(image, check_left_fitx[idx]) and check_range(image, check_right_fitx[idx]):
            y_bottom_idx = idx
            break
    # TODO: Checking that they have similar curvature
    # TODO: Checking that they are separated by approximately the right distance horizontally

    # Check if there is a cross between left line and right line
    for idx in range(y_top_idx, y_bottom_idx + 1, 1):

        if x_distance[idx] < 0:
            result = False
            break
    # Checking that they are roughly parallel
    if result == True and check_outlier(x_distance):
        result = False
    if result == False:
        left_line.detected = False
        left_line.best_fit = None
        right_line.detected = False
        right_line.best_fit = None
    return result


left_line = Line()
right_line = Line()


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image where lines are drawn on lanes)

    undist_img = undistort(image, mtx, dist)
    perspective_img = perspective_transform(undist_img, perspective_mat)
    binary_warped = pipeline(perspective_img)

    # Search left lane line
    # print('left line: {}'.format(left_line.detected))
    # print('right line: {}'.format(right_line.detected))
    # l_windows = []
    # r_windows = []
    # if left_line.detected:
    #     leftx, lefty = search_around_poly_v2(binary_warped, left_line.best_fit)
    #     leftx, lefty, l_windows = find_lane_pixels_v1(binary_warped, None, True)
    # else:
    #     leftx, lefty, l_windows = find_lane_pixels_v1(binary_warped, None, True)
    # # Search right lane line
    # if right_line.detected:
    #     rightx, righty = search_around_poly_v2(binary_warped, right_line.best_fit)
    #     rightx, righty, r_windows = find_lane_pixels_v1(binary_warped, None, False)
    # else:
    #     rightx, righty, r_windows = find_lane_pixels_v1(binary_warped, None, False)
    l_windows, r_windows = find_lane_pixels(
        binary_warped, left_line, right_line)
    left_line.update(l_windows)
    right_line.update(r_windows)
    # is_reliabie(left_line, l_windows, right_line, r_windows)

    # left_fitx, left_fity, left_poly = fit_poly(image.shape, leftx, lefty)
    # right_fitx, right_fity, right_poly = fit_poly(image.shape, rightx, righty)
    # Fit a second order polynomial to each with np.polyfit()
    # print("len(lefty): {}, len(leftx): {}, detected: {}".format(len(lefty), len(leftx), left_line.detected))
    # print("len(righty): {}, len(rightx): {}, detected: {}".format(len(righty), len(rightx), left_line.detected))

    if left_line.detected and right_line.detected:
        def get_center_shift(left_x, middle_x, xm_per_pix, real_road_width):
            curr_left_dis = (middle_x - left_x) * xm_per_pix
            return (real_road_width / 2) - curr_left_dis
        #         left_poly = np.polyfit(lefty, leftx, 2)
        #         right_poly = np.polyfit(righty, rightx, 2)

        # Update left lane line information
        #         left_line.update_curr_fit(left_poly)
        #         left_line.allx, left_line.ally = leftx, lefty
        # Update right lane line information
        #         right_line.update_curr_fit(right_poly)
        #         right_line.allx, right_line.ally = rightx, righty

        #         left_line.radius_of_curvature = measure_curvature_real(leftx, lefty)
        #         right_line.radius_of_curvature = measure_curvature_real(rightx, righty)

        #         left_line.windows = l_windows
        #         right_line.windows = r_windows

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        # Calc both polynomials using ploty, left_fit and right_fit
        left_fitx = left_line.best_fit[0] * ploty**2 + \
            left_line.best_fit[1] * ploty + left_line.best_fit[2]
        right_fitx = right_line.best_fit[0] * ploty**2 + \
            right_line.best_fit[1] * ploty + right_line.best_fit[2]

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(
            color_warp, perspective_mat_inv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

        # Show text information
        shift = get_center_shift(
            left_fitx[-1], image.shape[1] // 2, left_line.xm_per_pix, 3.7)
        info1 = 'L radius: {:>8.3f}m'.format(left_line.radius_of_curvature)
        info2 = 'R radius: {:>8.3f}m'.format(right_line.radius_of_curvature)
        info3 = 'Vehicle is {:>4.2f}m {} of center'.format(
            abs(shift), 'left' if shift >= 0 else 'right')
        cv2.putText(result, info1, (30, 30), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, info2, (30, 60), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, info3, (30, 90), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 255), 2, cv2.LINE_AA)

        color_warped = np.dstack(
            (binary_warped, binary_warped, binary_warped)) * 255
        empty_img = np.dstack((np.zeros_like(binary_warped), np.zeros_like(
            binary_warped), np.zeros_like(binary_warped)))
        lane_color_img = draw_lane_pixel(
            binary_warped, left_line, right_line, ploty)
        top_img = cv2.hconcat(
            [perspective_img, color_warped, lane_color_img, empty_img])
        top_img = cv2.resize(
            top_img, (image.shape[1], 180), interpolation=cv2.INTER_LINEAR)
        final_image = np.zeros(
            (result.shape[0] + top_img.shape[0], result.shape[1], 3), dtype=np.uint8)
        final_image[:top_img.shape[0]] = top_img
        final_image[top_img.shape[0]:] = result
        return final_image
    else:
        info = 'Fail to detect lane line'
        cv2.putText(image, info, (30, 30), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 255, 255), 1, cv2.LINE_AA)
        color_warped = np.dstack(
            (binary_warped, binary_warped, binary_warped)) * 255
        empty_img = np.dstack((np.zeros_like(binary_warped), np.zeros_like(
            binary_warped), np.zeros_like(binary_warped)))
        top_img = cv2.hconcat(
            [perspective_img, color_warped, empty_img, empty_img])
        top_img = cv2.resize(
            top_img, (image.shape[1], 180), interpolation=cv2.INTER_LINEAR)
        final_image = np.zeros(
            (image.shape[0] + top_img.shape[0], image.shape[1], 3), dtype=np.uint8)
        final_image[:top_img.shape[0]] = top_img
        final_image[top_img.shape[0]:] = image
        return final_image


def draw_histogram(binary_warped):
    out_img = np.copy(binary_warped)
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    pass
