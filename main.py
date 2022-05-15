import numpy as np
from scipy.optimize import linprog
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from configs import cfg

import bokeh.models as bm, bokeh.plotting as pl
from bokeh.io import output_notebook


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
        v = Gauss(A).zero_right()*(-1)
        W = Simplex(v, self.cfg["bnd"]).opt()
        return W


class Method:
    """

    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.FBA = FBA(self.cfg)
        self.A = Matrix(self.cfg["path_mrx"]).get()

    def nulling(self):
        flux = []
        point = []

        for i in range(self.A.shape[0]):
            B = self.A.copy()
            B[i] *= 0
            W = self.FBA.esm(B)
            flux.append(W["x"])
            point.append(f'{i}')
        return [flux, point]

    def view_psa(self):
        flux = self.nulling()[0]
        flux.append(self.FBA.esm(self.A)['x'])
        fba_psa = PCA(n_components=2).fit_transform(flux)
        plt.scatter(fba_psa[:, 0], fba_psa[:, 1], color="g")
        plt.show()


def draw_vectors(x, y, radius=10, alpha=0.25, color='blue',
                 width=1000, height=800, show=True, **kwargs):
    """ draws an interactive plot for data points with auxilirary info on hover """
    if isinstance(color, str): color = [color] * len(x)
    data_source = bm.ColumnDataSource({ 'x' : x, 'y' : y, 'color': color, **kwargs })
    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show: pl.show(fig)
    return fig


if __name__ == "__main__":

    retw = Method(cfg.config_fba).nulling()
    retw[1].append('orig')

    A = Matrix(cfg.config_fba['path_mrx']).get()
    orig_v = FBA(cfg.config_fba).esm(A)['x']


    # ---------- experiment ---------->
    # powfe = []
    # neighbors = np.sqrt(np.sum((retw[0] - orig_v)**2, 1)) <= 0.0000000001
    # index_neighbors = np.array(np.where(neighbors == True))[0]
    # for i in index_neighbors:
    #     powfe.append(int(retw[1][i]))
    #
    # retw[0].append(orig_v)
    #
    # print(powfe)
    # print(len(powfe))
    # print(A)
    #
    # A = np.delete(A, powfe, axis=0)
    #
    # print(A)
    #
    # test_with_zero = FBA(cfg.config_fba).esm(A)['x']
    # retw[0].append(test_with_zero)
    # retw[1].append('!!!')
    # <---------- !!! ----------


    retw[0].append(orig_v)

    fba_psa = PCA(n_components=2).fit_transform(retw[0])
    draw_vectors(fba_psa[:, 0], fba_psa[:, 1], token=retw[1], color = 'green')
    output_notebook()


    # plt.scatter(fba_psa[:, 0], fba_psa[:, 1], color="g")
    # plt.show()