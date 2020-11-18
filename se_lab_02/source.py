import numpy as np
from matplotlib import pyplot as plt


class Integrator:
    def __init__(self, matrix: np.matrix, step: float) -> None:

        self._matrix = matrix
        self._step = step
        self.first_point_flag = True

    def _system(self, y_vec: np.array) -> np.matrix:
        return y_vec.dot(self._matrix)

    def next_point(self, x: float, y_vec: np.array) -> dict:
        if self.first_point_flag:
            self.first_point_flag = False

            return {"x": x, "y": y_vec}

        k1 = self._system(y_vec)
        k2 = self._system(y_vec + (k1 * self._step / 3))
        k3 = self._system(y_vec + (k2 * self._step * 2 / 3))

        x_next = x + self._step
        y_vec_next = y_vec + self._step * ((1 / 4) * k1 + (3 / 4) * k3)

        return {"x": x_next, "y": y_vec_next}


def plot(**series):
    plt.plot(*series.values())
    plt.show()


def main():
    A = np.matrix([[0.0, 1.0], [-4.0, -0.4]], dtype=np.longdouble)

    step = 0.001
    x_min = 0
    x_max = np.pi + 0.015
    y_0 = np.matrix([[0, 1]], dtype=np.longdouble)

    integrator = Integrator(A, step)
    y_1 = np.array([], dtype=np.longdouble)
    y_2 = np.array([], dtype=np.longdouble)

    x_curr = x_min
    y_curr = y_0

    while x_curr < x_max:
        point = integrator.next_point(x_curr, y_curr)

        x_curr = point["x"]
        y_curr = point["y"]
        y_1 = np.append(y_1, point["y"].item((0, 0)))
        y_2 = np.append(y_2, point["y"].item((0, 1)))

    y_1 = np.append(y_1, y_1[0])
    y_2 = np.append(y_2, y_2[0])

    plot(f=y_1, y=y_2)


if __name__ == "__main__":
    main()
