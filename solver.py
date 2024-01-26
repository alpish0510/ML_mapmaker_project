import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy
from scipy.sparse.linalg import cg, LinearOperator

class MLsolver:
    """
    MLsolver class is designed to solve for a maximum likelihood solution of an input map.

    Parameters:
    - map (numpy.ndarray): Input map for which maximum likelihood solution needs to be computed.
    - nside (int): number of pixels per side in the output map. Default is 100.

    Attributes:
    - map (numpy.ndarray): Input map.
    - nside (int): Number of pixels per side in the output map.
    - nscan (int): Input map size (one side). Calculated as 4 * nside.
    - npix (int): Number of pixels in the map. Calculated as nside^2.

    Functions:
    1. tod_signal()
    2. get_PointingMatrix()
    3. noisemodel(index)
    4. fmul(f, x)
    5. ml_solver(iN)
    """

    def __init__(self, map, nside=100):
        """
        Constructor for the MLsolver class.

        Parameters:
        - map (numpy.ndarray): Input map for which maximum likelihood solution needs to be computed.
        - nside (int): number of pixels per side in the output map. Default is 100.
        """
        self.map = map
        self.nside = nside
        self.nscan = 4 * nside
        self.npix = nside**2

    def tod_signal(self):
        """
        Concatenates the map and its transpose to form the time-ordered data (TOD) signal.

        Returns:
        - numpy.ndarray: a TOD signal.
        """
        signal = np.concatenate([self.map.reshape(-1), self.map.T.reshape(-1)])
        return signal

    def get_PointingMatrix(self):
        """
        Generates the pointing matrix for the TOD signal.

        Returns:
        - scipy.sparse.csr_matrix: Pointing matrix.
        - int: Number of samples (nsamp).
        """
        pix_pat1 = (np.mgrid[:self.nscan, :self.nscan] * self.nside / self.nscan).reshape(2, -1)
        pix_pat2 = pix_pat1[::-1]  # swap x and y for the other pattern
        pix = np.concatenate([pix_pat1, pix_pat2], 1)  # Combine the two patterns
        nsamp = pix.shape[1]
        iy, ix = np.floor(pix + 0.5).astype(int) % self.nside
        P_nn = scipy.sparse.csr_matrix((np.full(nsamp, 1), (np.arange(nsamp), iy * self.nside + ix)),
                                       shape=(nsamp, self.npix))
        return P_nn, nsamp

    def noisemodel(self, index):
        """
        Defines the noise model based on the index of the slope.

        Parameters:
        - index (float): Index parameter for the noise model.

        Returns:
        - numpy.ndarray: Noise model (iN).
        """
        nsamp = self.get_PointingMatrix()[1]
        fknee = 0.01
        freq = np.fft.rfftfreq(nsamp)
        iN = 1 / (1 + (np.maximum(freq, freq[1] / 2) / fknee) ** index)
        iN = np.maximum(iN, np.max(iN) * 1e-8)
        return iN

    @staticmethod
    def fmul(f, x):
        """
        Performs the multiplication of a function with a signal in Fourier space.

        Parameters:
        - f (numpy.ndarray): The function.
        - x (numpy.ndarray): Input signal.

        Returns:
        - numpy.ndarray: Result of the multiplication.
        """
        if np.asarray(f).size == 1:
            return f * x
        else:
            return np.fft.irfft(np.fft.rfft(x) * f, n=len(x)).real

    def ml_solver(self, noise_model_index):
        """
        Solves for the maximum likelihood solution using a conjugate gradient solver.

        Parameters:
        - noise_model_index (float): Choose the slope of the noise model.

        Returns:
        - numpy.ndarray: Maximum likelihood solution.
        """
        P_nn = self.get_PointingMatrix()[0]
        tod = self.tod_signal()
        iN=self.noisemodel(noise_model_index)

        # Construct A as a LinearOperator
        def A_op(x):
            return P_nn.T.dot(self.fmul(iN, P_nn.dot(x)))

        b = P_nn.T.dot(self.fmul(iN, tod))
        A = LinearOperator((len(b), len(b)), matvec=A_op)
        x, info = cg(A, b, tol=1e-8)
        if info > 0:
            print("Conjugate gradient solver did not converge.")

        return x.reshape(self.nside, self.nside)


