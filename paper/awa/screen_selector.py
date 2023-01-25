import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy import optimize


class ScreenFinder:
    def __init__(self, data):
        self.data = data
        fig,ax = plt.subplots()
        self.ax = ax
        self.ax.imshow(data)

        self.points = [[0, 0]]
        self.line, = self.ax.plot(self.points[0][0], self.points[0][1], '+')
        self.circle = Circle((0.0, 0.0), 0.1, color='r', fill=False)
        self.ax.add_artist(self.circle)
        self.cid = self.ax.figure.canvas.mpl_connect('button_press_event',
                                                     self.draw_circle)

        plt.show()

    def draw_circle(self, event):
        # print('click', event)
        if event.inaxes != self.ax: return
        self.points.append((event.xdata, event.ydata))

        self.line.set_data(*np.asfarray(self.points).T)

        # draw a circle if the number of points is enough
        if len(self.points) > 3:
            points = self.points[1:]
            radius, center = self.fit_circle(points)
            self.circle.center = center
            self.circle.radius = radius

        self.ax.figure.canvas.draw()

    def calcuate_circle(self, points):
        # http://paulbourke.net/geometry/circlesphere/

        p1 = points[0]
        p2 = points[1]
        p3 = points[2]

        ma = (p2[1] - p1[1]) / (p2[0] - p1[0])
        mb = (p3[1] - p2[1]) / (p3[0] - p2[0])

        center_x = (ma * mb * (p1[1] - p3[1]) + mb * (p1[0] + p2[0]) - ma * (
                    p2[0] + p3[0])) / (2 * (mb - ma))
        center_y = -(1 / ma) * (center_x - (p1[0] + p2[0]) / 2) + (p1[1] + p2[1]) / 2

        radius = np.sqrt((p1[0] - center_x) ** 2 + (p1[1] - center_y) ** 2)

        return radius, (center_x, center_y)

    def fit_circle(self, points):
        # https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
        npts = np.asfarray(points).T
        x = npts[0]
        y = npts[1]
        x_m = np.mean(x)
        y_m = np.mean(y)

        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        def f_2(c):
            """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c[:2])
            return np.sum((Ri - c[-1])**2)

        x0 = np.array((x_m, y_m,100))
        print(x0)
        res = optimize.minimize(f_2, x0)

        return res.x[-1], res.x[:-1]

    def get_pixel_scale(self):
        return self.screen_diameter / self.circle.radius * 2


f = "D:\AWA\phase_space_tomography_run2\Quadscan_data\\0.npy"
data = np.load(f)
m = ScreenFinder(data[0])
print(m.circle)
