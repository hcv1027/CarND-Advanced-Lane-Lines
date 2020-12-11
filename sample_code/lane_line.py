import numpy as np


class Window():
    def __init__(self):
        self.x_low = None
        self.x_high = None
        self.y_low = None
        self.y_high = None
        self.allx = None
        self.ally = None

# Define a class to receive the characteristics of each line detection


class Line():
    def __init__(self):
        # The number of the last records we save
        self.record_size = 10
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations in real-world
        self.best_fit_real = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # search windows, for drawing
        self.windows = None
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 3 / 180  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 610  # meters per pixel in x dimension

    def update(self, windows):
        allx = None
        ally = None
        empty_counter = 0
        line_counter = 0
        new_line = False
        closest_line_idx = -1
        for idx in range(len(windows)):
            win = windows[idx]
            if win.allx.shape[0] == 0:
                empty_counter += 1
                new_line = False
            else:
                if closest_line_idx == -1:
                    closest_line_idx = idx
                if new_line == False:
                    new_line = True
                    line_counter += 1
            if allx == None:
                allx, ally = win.allx, win.ally
            else:
                allx = np.concatenate((allx, win.allx))
                ally = np.concatenate((ally, win.ally))
        best_weight = empty_counter / len(windows)
        # print('empty_counter: {}, new_line: {}'.format(empty_counter, new_line))
        update_best_fit = False if (
            empty_counter >= 6 and new_line < 2) else True
        # if update_best_fit == False:
        #    print('Skip best fit update')

        if empty_counter == len(windows) or closest_line_idx > nwindows // 3:
            self.detected = False
            self.best_fit = None
        else:
            self.detected = True

        # Update best_fit
        if self.detected:
            self.allx, self.ally = allx, ally
            self.radius_of_curvature = self.measure_curvature_real(allx, ally)
            self.windows = windows
            self.current_fit = np.polyfit(ally, allx, 2)
            if self.best_fit == None:
                self.best_fit = self.current_fit
            if update_best_fit:
                curr_fit = self.current_fit.reshape((1, 3))
                best_fit = self.best_fit.reshape((1, 3))
                self.best_fit = np.average(np.concatenate(
                    (curr_fit, best_fit), axis=0), axis=0, weights=[1.0 - best_weight, best_weight])

    def measure_curvature_real(self, plotx, ploty, y_eval=None):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Compute the polynomial in real world
        fit_cr = np.polyfit(ploty * self.ym_per_pix,
                            plotx * self.xm_per_pix, 2)

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty) if y_eval == None else y_eval

        # Compute the radius of curvature
        curverad = (1 + (2 * fit_cr[0] * y_eval * self.ym_per_pix +
                         fit_cr[1])**2)**1.5 / np.absolute(2 * fit_cr[0])

        return curverad
