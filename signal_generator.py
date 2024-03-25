import numpy as np, scipy
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, LinearOperator
from scipy.signal import welch

# Useful functions
def calc_l(n):
	ly = np.fft.fftfreq(n)[:,None]
	lx = np.fft.fftfreq(n)[None,:]
	l  = (ly**2+lx**2)**0.5
	return l

def fmul(f,x):
	if np.asarray(f).size == 1: return f*x
	else: return np.fft.irfft(np.fft.rfft(x)*f,n=len(x)).real

nside = 300   # Number of pixels per side
nscan = nside*4 # Number of samples per row. 
npix  = nside**2 # Number of pixels in map



# Generate two scanning patterns
pix_pat1 = (np.mgrid[:nscan,:nscan]*nside/nscan).reshape(2,-1)
pix_pat2 = pix_pat1[::-1] # swap x and y for other pattern
pix      = np.concatenate([pix_pat1,pix_pat2],1)  # Combine the two patterns
nsamp    = pix.shape[1]

# Build a nearest neighbor sparse pointing matrix
iy, ix  = np.floor(pix+0.5).astype(int)%nside  # Nearest pixel
P_nn    = scipy.sparse.csr_array((np.full(nsamp,1),(np.arange(nsamp),iy*nside+ix)),shape=(nsamp,npix))


# Creating a noise simulator
def simul_noise(fknee, alpha, N_white):
    """
    Simulates noise based on knee frequency (fknee) and the index of the power law (alpha).

    Parameters:
    fknee (float): The knee frequency.
    alpha (float): The spectral index.
    nsamp (int): The number of samples to generate.

    Returns:
    noise (numpy.ndarray): The generated noise signal.
    iN (numpy.ndarray): The inverse of the noise power spectrum.

    The function first calculates the frequency array for the given number of samples using np.fft.rfftfreq.
    It then calculates the power spectrum using the given fknee and alpha parameters.
    The power spectrum is then inverted and limited to a minimum value to avoid division by zero.
    Finally, the function generates a noise sample by multiplying a random sample from a standard normal distribution with the square root of the inverted power spectrum.
    """
    freq  = np.fft.rfftfreq(nsamp)
    iN    = 1/(N_white*(1+(np.maximum(freq,freq[1]/2)/fknee)**alpha))
    iN    = np.maximum(iN, np.max(iN)*1e-8)
    noise  = fmul(iN**-0.5, np.random.standard_normal(nsamp))
    return noise, iN

# Creating a signal simulator
def simul_signal(nscan, bsigma, index, C=None):
    """
    Simulates a signal based on a given power spectrum.

    Parameters:
    nscan (int): The number of scans, which determines the size of the 2D array to generate.
    bsigma (float): The standard deviation of the Gaussian beam window function.
    index (float): The spectral index for calculating the power spectrum.
    C (numpy.ndarray, optional): An optional input power spectrum. If not provided, a power spectrum is calculated based on the other parameters.

    Returns:
    signal (numpy.ndarray): The simulated signal as a 1D array.
    signal_map (numpy.ndarray): The simulated signal as a 2D array.

    The function first calculates the spatial frequencies associated with the Fourier transform of a 2D array of size nscan x nscan. If a power spectrum is not provided, it calculates one based on these spatial frequencies, a normalization factor, and the given spectral index. It then calculates a Gaussian beam window function based on the spatial frequencies and the given beam standard deviation. Next, it generates a random 2D array from a standard normal distribution, performs a Fourier transform on it, multiplies the result with the square root of the power spectrum and the beam window function, performs an inverse Fourier transform on the result, and extracts the real part. Finally, it reshapes the 2D array into a 1D array and returns both the 1D and 2D arrays.
    """
    l = calc_l(nscan)*nscan/nside # express in units of output pixels
    lnorm = 1
    if C is None:
        C = (np.maximum(l, l[0, 1]/2)/lnorm)**index
    B = np.exp(-0.5*l**2*bsigma**2)
    signal_map = np.fft.ifft2(np.fft.fft2(np.random.standard_normal((nscan, nscan)))*C**0.5*B).real
    signal = np.concatenate([signal_map.reshape(-1), signal_map.T.reshape(-1)])
    return signal, signal_map

# Calculating the PSD
def calculate_psd(signal, noise):
    """
    Calculates the power spectral density (PSD) of a signal, noise, and their sum.

    Parameters:
    signal (numpy.ndarray): The input signal.
    noise (numpy.ndarray): The input noise.

    Returns:
    (f_s, psd_s) (tuple): The frequencies and PSD of the signal.
    (f_n, psd_n) (tuple): The frequencies and PSD of the noise.
    (f_ns, psd_ns) (tuple): The frequencies and PSD of the sum of the signal and noise.

    The function uses the Welch method to estimate the PSD of the signal, noise, and their sum. The PSD is calculated with a Hanning window and a segment length of 1024.
    """
    f_s, psd_s = welch(signal, fs=1, nperseg=1024, window='hanning')
    f_n, psd_n = welch(noise, fs=1, nperseg=1024, window='hanning')
    f_ns, psd_ns = welch(noise+signal, fs=1, nperseg=1024, window='hanning')
    return (f_s, psd_s), (f_n, psd_n), (f_ns, psd_ns)

def plot_psd(psd_signal, psd_noise, psd_signal_noise, keyword):
    """
    Plots the power spectral density (PSD) of a signal, noise, and their sum.

    Parameters:
    psd_signal (tuple): A tuple containing the frequencies and PSD of the signal.
    psd_noise (tuple): A tuple containing the frequencies and PSD of the noise.
    psd_signal_noise (tuple): A tuple containing the frequencies and PSD of the sum of the signal and noise.
    keyword (str): A string to be included in the title of the plot to indicate the type of noise.

    The function creates three subplots. The first subplot is a log-log plot of the signal PSD. The second subplot is a log-log plot of the noise PSD. The third subplot is a log-log plot of the PSD of the sum of the signal and noise, with the signal and noise PSDs also plotted for comparison. The function also sets the title, x-label, and y-label for each subplot, and adds a legend to the third subplot. Finally, the function sets the title for the entire figure and displays the plot.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    ax[0].loglog(*psd_signal)
    ax[0].set_title('Signal PSD', fontsize=15)
    ax[0].set_xlabel('Frequency', fontsize=15)
    ax[0].set_ylabel('PSD', fontsize=15)
    
    ax[1].loglog(*psd_noise)
    ax[1].set_title('Noise PSD', fontsize=15)
    ax[1].set_xlabel('Frequency', fontsize=15)
    ax[1].set_ylabel('PSD', fontsize=15)
    
    ax[2].loglog(*psd_signal_noise, label ='Signal+Noise',lw=2)
    ax[2].loglog(*psd_noise, alpha=0.5, label='Noise', c='r', ls='--')
    ax[2].loglog(*psd_signal, alpha=0.5, label='Signal', c='g', ls='--')
    ax[2].set_title('Signal+Noise PSD', fontsize=15)
    ax[2].set_xlabel('Frequency', fontsize=15)
    ax[2].set_ylabel('PSD', fontsize=15)
    ax[2].legend(fontsize=15)
    
    for a in ax:
        a.tick_params(axis='both', which='both', direction='in', labelsize=12)
        for spine in a.spines.values():
            spine.set_linewidth(1.5)
    
    fig.suptitle(f'Simulated CMB Signal and {keyword} Noise Power Spectral Density Analysis', fontsize=20)
    plt.tight_layout()
    plt.show()



