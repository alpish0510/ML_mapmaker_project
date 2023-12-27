import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, LinearOperator

class ML_map2d:
    """
    ML_map2d - 2D Maximum Likelihood Map-Making Class

    This class implements a 2D map-making algorithm based on Maximum Likelihood estimation.
    It includes methods for generating scanning patterns, building signal and noise models,
    simulating signal and noise, and performing Maximum Likelihood map-making.

    Attributes:
        nside (int): Number of pixels per side in the output map.
        nscan (int): Number of samples per row in the scanning pattern.
        npix (int): Total number of pixels in the output map.
        pix (numpy.ndarray): Scanning pattern coordinates.
        nsamp (int): Total number of samples in the scanning pattern.

    Methods:
        scanning_pattern(self) -> Tuple[numpy.ndarray, int]:
            Generates scanning patterns and returns the pixel coordinates and the total number of samples.

        get_pointing_matrix(self) -> scipy.sparse.csr_matrix:
            Returns the sparse pointing matrix based on the scanning pattern.

        noise_model(self) -> numpy.ndarray:
            Computes the noise model for the scanning pattern.

        build_signal(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
            Builds the signal model with a 1/l^2 spectrum and applies band-limiting.

        calc_l(self, n: int) -> numpy.ndarray:
            Helper method to calculate the 2D spatial frequency.

        fmul(self, f, x) -> numpy.ndarray:
            Multiply a function or array in Fourier space with a given signal.

        sim_signal(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
            Simulates the signal map and returns flattened signal and the signal map.

        sim_noise(self) -> numpy.ndarray:
            Simulates noise based on the noise model.

        ml_map(self, x: numpy.ndarray) -> numpy.ndarray:
            Performs Maximum Likelihood map-making given a simulated signal and returns the map.

        plotter(self) -> None:
            Plots the ML map and the noise map side by side.

    Example:
    >>>    ml_obj = ML_map2d(nside=100)
    >>>    ml_obj.plotter()

    """

    

    def __init__(self, nside):
        """
        Initializes the ML_map2d object with the specified number of pixels per side (nside).

        Args:
            nside (int): Number of pixels per side in the output map.
        """
        self.nside = nside
        self.nscan = 4 * nside
        self.npix = nside**2
        self.pix = None
        self.nsamp = None



    def scanning_pattern(self):
        """
    Generate a 2D scanning pattern for the map.

    Returns:
    - numpy.ndarray: Array containing the pixel coordinates of the scanning pattern.
    - int: Total number of samples in the scanning pattern.

    The scanning pattern is created by generating a 2D grid with dimensions
    determined by `self.nscan`. The grid is then rescaled to match the desired
    map size (`self.nside`). Two patterns are created by swapping the x and y
    coordinates, and they are concatenated to form the final scanning pattern.

    Example:
    >>> pix, nsamp = ml_map2d_instance.scanning_pattern()
    """
        
        pix_pat1 = (np.mgrid[:self.nscan, :self.nscan] * self.nside / self.nscan).reshape(2, -1)
        pix_pat2 = pix_pat1[::-1]  # swap x and y for the other pattern
        self.pix = np.concatenate([pix_pat1, pix_pat2], 1)
        self.nsamp = self.pix.shape[1]

        return self.pix, self.nsamp



    
    def get_pointing_matrix(self):
        """
    Generate a pointing matrix based on the current scanning pattern.

    Returns:
    - scipy.sparse.csr_matrix: Pointing matrix representing the mapping from
      samples in the scanning pattern to pixels on the map.

    The pointing matrix is computed by converting the pixel coordinates from
    the scanning pattern (`self.pix`) to pixel indices on the map (`self.nside`).
    The resulting matrix has dimensions (`self.nsamp`, `self.npix`), where each
    row corresponds to a sample in the scanning pattern, and columns correspond
    to pixels on the map.

    Example:
    >>> P_nn = ml_map2d_instance.get_pointing_matrix()
    """
        if self.pix is None or self.nsamp is None:
            self.scanning_pattern()
        iy, ix = np.floor(self.pix + 0.5).astype(int) % self.nside
        P_nn = sp.csr_matrix((np.full(self.nsamp, 1), (np.arange(self.nsamp), iy * self.nside + ix)),
                             shape=(self.nsamp, self.npix))

        return P_nn




    def noise_model(self):
        """
    Generate a noise power spectrum for the scanning pattern.

    Returns:
    - numpy.ndarray: Noise power spectrum for the scanning pattern.

    The noise model is computed based on the power spectrum of the scanning
    pattern. The knee frequency is determined by the relation to the scan size,
    and the resulting power spectrum is modified to ensure stability and avoid
    division by zero.

    Example:
    >>> iN = ml_map2d_instance.noise_model()
    """
        if self.nsamp is None:
            _, nsamp = self.scanning_pattern()
        else:
            nsamp = self.nsamp

        fknee = 0.5 * 30 / self.nscan
        freq = np.fft.rfftfreq(nsamp)
        iNo = 1 / (1 + (np.maximum(freq, freq[1] / 2) / fknee)**-3.5)
        iN = np.maximum(iNo, np.max(iNo)*1e-8)
        
        return iN
    



    def build_signal(self):
        """
    Build a signal with a simple 1/l^2 spectrum and apply band-limiting.

    Returns:
    - Tuple[numpy.ndarray, numpy.ndarray]: Tuple containing the signal power
      spectrum (C) and the band-limiting function (B).

    This function constructs a signal with a power spectrum following a 1/l^2
    pattern. The signal is further band-limited by applying a beam function.

    Example:
    >>> C, B = ml_map2d_instance.build_signal()
    """
        # Build the signal with a simple 1/l**2 spectrum
        l = self.calc_l(self.nscan) * self.nscan / self.nside
        lnorm = 1
        C = (np.maximum(l, l[0, 1] / 2) / lnorm)**-2

        # Apply band-limiting by using a beam
        bsigma = 3
        B = np.exp(-0.5 * l**2 * bsigma**2)

        return C, B
    



    @staticmethod
    def calc_l(n):
        """
    Calculate the radial distance in Fourier space for a given grid size.

    Args:
    - n (int): Grid size.

    Returns:
    numpy.ndarray: Radial distance in Fourier space.

    This static method computes the radial distance in Fourier space for a given
    grid size 'n'. It returns a 2D array representing the radial distances.

    Example:
    >>> l = ML_map2d.calc_l(64)
    """
        ly = np.fft.fftfreq(n)[:, None]
        lx = np.fft.fftfreq(n)[None, :]
        l = (ly**2 + lx**2)**0.5
        return l
    



    @staticmethod
    def fmul(f, x):
        """
    Multiply a function or array in Fourier space with a given signal.

    Args:
    - f (Union[float, numpy.ndarray]): Function or array in Fourier space.
    - x (numpy.ndarray): Input signal.

    Returns:
    numpy.ndarray: Result of the multiplication in real space.

    This static method multiplies a function or array 'f' in Fourier space with
    the input signal 'x'. It returns the result in real space.

    Example:
    >>> result = ML_map2d.fmul(2.0, signal)
    """
        if np.asarray(f).size == 1:
            return f * x
        else:
            return np.fft.irfft(np.fft.rfft(x) * f, n=len(x)).real
        


        
    
    # def A(self):
    #     P=self.get_pointing_matrix()
    #     iN, _ = self.noise_model()
    #     x,_= self.sim_signal()
    #     res = P.T.dot(self.fmul(iN, P.dot(x)))
    #     return res

    def sim_signal(self):
        """
    Simulate a signal map and return the flattened signal.

    Returns:
    Tuple[numpy.ndarray, numpy.ndarray]:
        Flattened signal and the original signal map.

    This method simulates a signal map by applying a Fourier-space spectrum
    defined by the build_signal method. It returns both the flattened signal
    and the original signal map.

    Example:
    >>> signal, signal_map = ML_map2d.sim_signal()
    """
        C, B = self.build_signal()
        signal_map = np.fft.ifft2(np.fft.fft2(np.random.standard_normal((self.nscan, self.nscan))) * C**0.5 * B).real
        signal = np.concatenate([signal_map.reshape(-1), signal_map.T.reshape(-1)])

        return signal, signal_map
    



    def sim_noise(self):
        """
    Simulate noise and return the flattened noise.

    Returns:
    numpy.ndarray: Flattened noise.

    This method simulates noise by applying a Fourier-space noise model defined
    by the noise_model method. It returns the flattened noise.

    Example:
    >>> noise = ML_map2d.sim_noise()
    """
        _, nsamp = self.scanning_pattern()
        iN = self.noise_model()
        noise = self.fmul(iN**-0.5, np.random.standard_normal(nsamp))

        return noise
    


    
    def ml_map(self, x):
        """
    Perform maximum likelihood map-making.

    Parameters:
    x (numpy.ndarray): input tod data.

    Returns:
    numpy.ndarray: Reshaped ML map.

    This method performs maximum likelihood (ML) map-making using the
    Conjugate Gradient (CG) solver from SciPy. It constructs a linear operator
    A, defines the right-hand side of the linear system, and solves it to obtain
    the ML map.

    Example:
    >>> ml_map_result = ML_map2d.ml_map(data)
    """
        P_nn = self.get_pointing_matrix()
        signal, _ = self.sim_signal()
        noise = self.noise_model()

        # Construct A as a LinearOperator
        def A_op(x):
            return P_nn.T.dot(self.fmul(noise, P_nn.dot(x)))

        # Right-hand side of the linear system
        b = P_nn.T.dot(self.fmul(noise, x))

        # Construct the LinearOperator directly in the ml_map function
        A = LinearOperator((len(b), len(b)), matvec=A_op)

        # Solve the linear system using SciPy's CG solver
        x, info = cg(A, b, tol=1e-8)

        if info > 0:
            print("Conjugate gradient solver did not converge.")

        return x.reshape(self.nside, self.nside)
    




    def plotter(self):
        """
    Plot ML map and noise map side by side.

    This method generates a 1x2 subplot, where the first subplot displays
    the ML map, and the second subplot displays the noise map. Colorbars are
    added to each subplot for reference.

    Example:
    >>> map_instance = ML_map2d(nside)
    >>> map_instance.plotter()
    """
        signal, _ = self.sim_signal()
        noise = self.sim_noise()
        mlmap = self.ml_map(signal)
        noise = self.ml_map(noise)
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].imshow(mlmap)
        ax[1].imshow(noise)
        ax[0].set_title("ML map",fontsize=15)
        ax[1].set_title("Noise map",fontsize=15)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        fig.colorbar(ax[0].imshow(mlmap), ax=ax[0])
        fig.colorbar(ax[1].imshow(noise), ax=ax[1])
        plt.tight_layout()
        plt.show()


        

    
# map1=ML_map2d(100)
# #plt.plot(np.arange(320000),map1.sim_noise())
# #print(map1.noise_model()[0].shape)
# plt.imshow(map1.ml_map())
# plt.show()

