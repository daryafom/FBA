import numpy as np
from scipy.optimize import linprog
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from configs import cfg


class Matrix:
    """
    """

    def __init__(self, file_path):
        with open(file_path, "r") as f:
            matrix_file = f.read()
        lines = matrix_file.split("\n")
        self.matrix = np.array([i.split() for i in lines], dtype=np.double)

    def get(self):
        """
        :return:
        """
        return self.matrix


class MatrixError(Exception):
    """
    Input Matrix Error Handler
    """


class Gauss:
    """
    Method's of Gauss
    """

    def __init__(self, A):

        self.line = A.shape[0]
        self.columns = A.shape[1]
        self.A = A.copy()

    def forward(self):
        """
        :return:
        """
        GaussForward = self.A.copy()

        step = 0
        while step < self.line:
            if GaussForward[step][step] == 0:
                for idx_line in range(step + 1, self.line):
                    if GaussForward[idx_line, step] != 0:
                        line = GaussForward[idx_line].copy()
                        GaussForward[idx_line] = GaussForward[step].copy()
                        GaussForward[step] = line.copy()
                        break
                else:
                    step += 1

            if step == self.line:
                break

            GaussForward[step] /= GaussForward[step][step]
            for idx_line in range(step + 1, self.line):
                GaussForward[idx_line] -= (
                    GaussForward[idx_line][step] * GaussForward[step]
                )
            step += 1

        return GaussForward

    def zero_right(self):
        """
        :return:
        """
        if self.columns == self.line:
            raise MatrixError("error matrix")
        elif self.columns < self.line:
            raise MatrixError("error matrix")

        A = self.forward()
        v = np.zeros(self.columns)
        v[self.line :] = 1

        for i in range(self.line - 1, -1, -1):
            for j in range(self.line - i):
                v[i] += A[i, self.columns - (j + 1)] * v[self.line - j]
            v[i] *= -1

        return v


class Simplex:
    """

    """

    def __init__(self, obj, bnd, method="revised simplex"):
        self.obj = obj
        self.bnd = bnd
        self.method = method

    def opt(self):
        return linprog(c=self.obj, bounds=self.bnd, method=self.method)


class FBA:
    """

    """

    def __init__(self, cfg):
        self.cfg = cfg

    def esm(self, A):
        v = Gauss(A).zero_right()
        W = Simplex(v, self.cfg["bnd"]).opt()
        return W


class Method:
    """

    """

    def __init__(self, cfg):
        self.cfg = cfg

    def nulling(self):
        flux = []
        A = Matrix(self.cfg["path_mrx"]).get()
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                B = A.copy()
                B[i, j] = 0
                W = FBA(self.cfg).esm(B)
                flux.append(W["x"])
        return flux

    def view_psa(self):
        flux = self.nulling()
        fba_psa = PCA(n_components=2).fit_transform(flux)
        plt.scatter(fba_psa[:, 0], fba_psa[:, 1], color="g")
        plt.show()


if __name__ == "__main__":

    Method(cfg.config_fba).view_psa()
