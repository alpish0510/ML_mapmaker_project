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

    def __init__(self, map, nside=100, N_white=1.8, fknee=0.01):
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
        self.N_white = N_white
        self.fknee = fknee

    

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
        N_white = self.N_white
        fknee = self.fknee
        freq = np.fft.rfftfreq(nsamp)
        iN = 1 / (N_white*(1 + (np.maximum(freq, freq[1] / 2) / fknee) ** index))
        iN = np.maximum(iN, np.max(iN) * 1e-8)
        if index==0:
            noise  = self.fmul(iN**-0.5, np.random.normal(0,1,nsamp))
        else:
            noise  = self.fmul(iN**-0.5, np.random.standard_normal(nsamp))
        return iN, noise
    
    def tod_signal(self,index,with_noise=True):
        """
        Concatenates the map and its transpose to form the time-ordered data (TOD) signal.

        Returns:
        - numpy.ndarray: a TOD signal.
        """
        
        if with_noise==True:
            signal=np.concatenate([self.map.reshape(-1), self.map.T.reshape(-1)]) + self.noisemodel(index)[1]
        else:
            signal = np.concatenate([self.map.reshape(-1), self.map.T.reshape(-1)])
            
        return signal

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
        
    def map_w_noise(self,noise_model_index):
        tod = self.tod_signal(noise_model_index)
        noisy_map=tod[:tod.size//2].reshape(self.nscan, self.nscan)
        return noisy_map
        


    def ml_solver(self, noise_model_index, add_noise=True):
        """
        Solves for the maximum likelihood solution using a conjugate gradient solver.

        Parameters:
        - noise_model_index (float): Choose the slope of the noise model.

        Returns:
        - numpy.ndarray: Maximum likelihood solution.
        """
        P_nn = self.get_PointingMatrix()[0]
        if add_noise==True:
            tod = self.tod_signal(noise_model_index)
        else:
            tod = self.tod_signal(noise_model_index,with_noise=False)
        iN=self.noisemodel(noise_model_index)[0]

        # Construct A as a LinearOperator
        def A_op(x):
            return P_nn.T.dot(self.fmul(iN, P_nn.dot(x)))

        b = P_nn.T.dot(self.fmul(iN, tod))
        A = LinearOperator((len(b), len(b)), matvec=A_op)
        x, info = cg(A, b, tol=1e-8)
        if info > 0:
            print("Conjugate gradient solver did not converge.")

        return x.reshape(self.nside, self.nside)
    

def cosine_window(N):
    "makes a cosine window for apodizing to avoid edges effects in the 2d FFT" 
    # make a 2d coordinate system
    N=int(N) 
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.)/N *np.pi ## eg runs from -pi/2 to pi/2
    X = np.outer(ones,inds)
    Y = np.transpose(X)
  
    # make a window map
    window_map = np.cos(X) * np.cos(Y)
   
    # return the window map
    return(window_map)


def ps_calc(input_map,ang_size,ell_max=5000,delta_ell=50,appodize=True):
    N=int(input_map.shape[0])
    pix_size=ang_size*60/N
    if appodize==True:
        Map1, Map2 = cosine_window(N)*input_map, cosine_window(N)*input_map
    else:
        Map1, Map2 = input_map, input_map
    # make a 2d ell coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) /(N-1.)
    kX = np.outer(ones,inds) / (pix_size/60. * np.pi/180.)
    kY = np.transpose(kX)
    K = np.sqrt(kX**2. + kY**2.)
    ell_scale_factor = 2. * np.pi 
    ell2d = K * ell_scale_factor
    
    # make an array to hold the power spectrum results
    N_bins = int(ell_max/delta_ell)
    ell_array = np.arange(N_bins)
    CL_array = np.zeros(N_bins)
    
    # get the 2d fourier transform of the map
    FMap1 = np.fft.ifft2(np.fft.fftshift(Map1))
    FMap2 = np.fft.ifft2(np.fft.fftshift(Map2))
    PSMap = np.fft.fftshift(np.real(np.conj(FMap1) * FMap2))
    # fill out the spectra
    i = 0
    while (i < N_bins):
        ell_array[i] = (i + 0.5) * delta_ell
        inds_in_bin = ((ell2d >= (i* delta_ell)) * (ell2d < ((i+1)* delta_ell))).nonzero()
        CL_array[i] = np.mean(PSMap[inds_in_bin])
        #print i, ell_array[i], inds_in_bin, CL_array[i]
        i = i + 1
    # the binned spectrum
    binned_spectrum=CL_array*np.sqrt(pix_size /60.* np.pi/180.)*2.

    # return the power spectrum and ell bins
    return(ell_array,(binned_spectrum* ell_array * (ell_array+1.)/2. / np.pi))
    

