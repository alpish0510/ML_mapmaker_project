import numpy as np
import scipy
from scipy.signal import welch
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse.linalg import cg, LinearOperator
import solver as slv
import argparse
import os
import time

# Marking the start time of the code
start_time = time.time()

# Creating a directory to save the output files
directory ='Noise Analysis'
if not os.path.exists(directory):
    os.makedirs(directory)

#creating the parser for the script
parser=argparse.ArgumentParser(description='Simulate a CMB map and perform a signal analysis')
parser.add_argument("nside", type=int, help='Number of pixels per side of the output map')
parser.add_argument("output", type=str, help='Name of the output file')
parser.add_argument("fknee", type=float, help='knee frequency of the noise (Hz)')
parser.add_argument("alpha", type=float, help='spectral index of the noise')
parser.add_argument("realistic_CMB", type=int, help='to simulate a realistic CMB map or not (0 for no, 1 for yes)')
parser.add_argument("--N_w", type=float, default="1.8",help='white noise level (default is 1.8 in units of 10^-5 uK^2)')

#parsing the arguments
args=parser.parse_args()

#assigning the arguments to variables
nside = args.nside
output = args.output
fknee = args.fknee
alpha = args.alpha
realistic_CMB = args.realistic_CMB
N_white = args.N_w

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

def fmul(f,x):
	if np.asarray(f).size == 1: return f*x
	else: return np.fft.irfft(np.fft.rfft(x)*f,n=len(x)).real
     
def calc_l(n):
    ly = np.fft.fftfreq(n)[:,None]
    lx = np.fft.fftfreq(n)[None,:]
    l  = (ly**2+lx**2)**0.5
    return l


# Creating a noise simulator
def simul_noise(fknee, alpha, nsamp, N_white=1):
    """
    Simulates noise based on knee frequency (fknee) and the index of the power law (alpha).

    Parameters:
    fknee (float): The knee frequency.
    alpha (float): The spectral index.
    nsamp (int): The number of samples to generate.

    Returns:
    noise (numpy.ndarray): The generated noise signal.
    iN (numpy.ndarray): The inverse noise model.
    N_white (float): The white noise level. Default is 1. The units are 10^-5 uK^2.

    The function first calculates the frequency array for the given number of samples using np.fft.rfftfreq.
    It then calculates the noise model using the given fknee and alpha parameters.
    The noise model is then inverted and limited to a minimum value to avoid division by zero.
    Finally, the function generates a noise sample by multiplying a random sample from a standard normal distribution with the square root of the inverted power spectrum.
    """
    freq  = np.fft.rfftfreq(nsamp)
    iN    = 1/(N_white*(1+(np.maximum(freq,freq[1]/2)/fknee)**alpha))
    iN    = np.maximum(iN, np.max(iN)*1e-8)
    if alpha == 0:
        noise  = fmul(iN**-0.5,np.random.normal(0,1,nsamp))
    else:
        noise  = fmul(iN**-0.5, np.random.standard_normal(nsamp))
    return noise, iN

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
def calculate_psd(signal, noise, nperseg=1024):
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
    f_s, psd_s = welch(signal, fs=1, nperseg=nperseg, window='hamming')
    f_n, psd_n = welch(noise, fs=1, nperseg=nperseg, window='hamming')
    f_ns, psd_ns = welch(noise+signal, fs=1, nperseg=nperseg, window='hamming')
    return (f_s, psd_s), (f_n, psd_n), (f_ns, psd_ns)

def plot_psd(psd_signal, psd_noise, psd_signal_noise, keyword, modif=False):
    """
    Plots the power spectral density (PSD) of a signal, noise, and their sum.

    Parameters:
    psd_signal (tuple): A tuple containing the frequencies and PSD of the signal.
    psd_noise (tuple): A tuple containing the frequencies and PSD of the noise.
    psd_signal_noise (tuple): A tuple containing the frequencies and PSD of the sum of the signal and noise.
    keyword (str): A string to be included in the title of the plot.

    The function creates three subplots. The first subplot is a log-log plot of the signal PSD. The second subplot is a log-log plot of the noise PSD. The third subplot is a log-log plot of the PSD of the sum of the signal and noise, with the signal and noise PSDs also plotted for comparison. The function also sets the title, x-label, and y-label for each subplot, and adds a legend to the third subplot. Finally, the function sets the title for the entire figure and displays the plot.
    """
    if modif==False:
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
        ax[2].legend(fontsize=11)
        
        for a in ax:
            a.tick_params(axis='both', which='both', direction='in', labelsize=12)
            for spine in a.spines.values():
                spine.set_linewidth(1.5)
        
        fig.suptitle(f'Simulated CMB Signal and {keyword} Noise Power Spectral Density Analysis', fontsize=20)
        plt.tight_layout()

    else:
        plt.style.use('default')
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        
        ax[0].loglog(*psd_signal)
        ax[0].set_title('Signal PSD', fontsize=15)
        ax[0].set_xlabel('Frequency (Hz)', fontsize=15)
        ax[0].set_ylabel(r'PSD ($\mu K^2$/Hz)', fontsize=15)
        
        ax[1].loglog(*psd_noise)
        ax[1].set_title('Noise PSD', fontsize=15)
        ax[1].set_xlabel('Frequency (Hz)', fontsize=15)
        ax[1].set_ylabel(r'PSD ($\mu K^2$/Hz)', fontsize=15)
        
        ax[2].loglog(*psd_signal_noise, label ='Signal+Noise',lw=2)
        ax[2].loglog(*psd_noise, alpha=0.5, label='Noise', c='r', ls='--')
        ax[2].loglog(*psd_signal, alpha=0.5, label='Signal', c='g', ls='--')
        ax[2].set_title('Signal+Noise PSD', fontsize=15)
        ax[2].set_xlabel('Frequency (Hz)', fontsize=15)
        ax[2].set_ylabel(r'PSD ($\mu K^2$/Hz)', fontsize=15)
        ax[2].legend(fontsize=11)
        
        for a in ax:
            a.tick_params(axis='both', which='both', direction='in', labelsize=12)
            for spine in a.spines.values():
                spine.set_linewidth(1.5)
        
        fig.suptitle(f'Simulated CMB Signal and {keyword} Noise Power Spectral Density Analysis', fontsize=20)
        plt.tight_layout()
    
def mapmaker_ml(tod, P, iN):
    b = P.T.dot(fmul(iN, tod))

    def A(x):
        return P.T.dot(fmul(iN, P.dot(x)))

    # Ensuring that A and b are compatible with scipy.sparse.linalg.cg
    A_op = LinearOperator((len(b), len(b)), matvec=A)

    # Solve the linear system using SciPy's CG solver
    x, info = cg(A_op, b, tol=1e-8)

    if info > 0:
        print("Conjugate gradient solver did not converge.")

    return x.reshape(nside, nside)


if realistic_CMB == 0:
    # Simulating the CMB signal
    signal,signal_map=simul_signal(nscan, 3, -2)
    # Simulating the red noise
    r_noise,r_iN=simul_noise(fknee, alpha, nsamp, N_white)
    # Simulating the white noise
    w_noise,w_iN=simul_noise(fknee, 0, nsamp, N_white)

    #plotting the signal, signal map, noise signals and signal + noises
    fig,ax=plt.subplots(3,2,figsize=(15,15))
    ax[0,0].plot(np.arange(len(signal)),signal)
    ax[0,0].set_xlabel("Sample", fontsize=14)
    ax[0,0].set_ylabel("Amplitude", fontsize=14)
    ax[0,0].set_title("Signal", fontsize=14)
    ax[0,1].imshow(signal_map,cmap="RdBu")
    ax[0,1].set_title("Signal Map", fontsize=14)
    ax[1,0].plot(np.arange(len(r_noise)),r_noise)
    ax[1,0].set_xlabel("Sample", fontsize=14)
    ax[1,0].set_ylabel("Amplitude", fontsize=14)
    ax[1,0].set_title(f"Red Noise ($\\alpha = {alpha}$)", fontsize=14)
    ax[1,1].plot(np.arange(len(w_noise)),w_noise)
    ax[1,1].set_xlabel("Sample", fontsize=14)
    ax[1,1].set_ylabel("Amplitude", fontsize=14)
    ax[1,1].set_title(r"White Noise ($\alpha = 0$)", fontsize=14)
    ax[2,0].plot(np.arange(len(signal)),signal+r_noise)
    ax[2,0].set_xlabel("Sample", fontsize=14)
    ax[2,0].set_ylabel("Amplitude", fontsize=14)
    ax[2,0].set_title(f"Signal+Red Noise ($\\alpha = {alpha}$)", fontsize=14)
    ax[2,1].plot(np.arange(len(signal)),signal+w_noise)
    ax[2,1].set_xlabel("Sample", fontsize=14)
    ax[2,1].set_ylabel("Amplitude", fontsize=14)
    ax[2,1].set_title(r'Signal+White Noise ($\alpha = 0$)', fontsize=14)
    cbar1=fig.colorbar(ax[0,1].imshow(signal_map,cmap="RdBu"), ax=ax[0,1])
    cbar1.set_label(r'Amplitude', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{directory}/{output}_simulated_signal_noise_maps.png')

    if nside==100:
        nperseg=1024
    elif nside==200:
        nperseg=2048
    else:
        nperseg=4096
    # Calculating the signal and red noise PSD
    psd_signal_red, psd_noise_red, psd_signal_noise_red = calculate_psd(signal, r_noise, nperseg=nperseg)
    # Calculating the signal and white noise PSD
    psd_signal_white, psd_noise_white, psd_signal_noise_white = calculate_psd(signal, w_noise, nperseg=nperseg)
    # Plotting the PSD
    plot_psd(psd_signal_red, psd_noise_red, psd_signal_noise_red, 'Red')
    plt.savefig(f'{directory}/{output}_simulated_signal_red_noise_psd.png')
    plot_psd(psd_signal_white, psd_noise_white, psd_signal_noise_white, 'White')
    plt.savefig(f'{directory}/{output}_simulated_signal_white_noise_psd.png')

    # ML solutions of the signals
    ml_s_w=mapmaker_ml(signal+w_noise,P_nn,w_iN)
    ml_s_r=mapmaker_ml(signal+r_noise,P_nn,r_iN)

    # Plotting the ML solutions
    fig,ax=plt.subplots(1,3,figsize=(17,5))
    ax[2].imshow(ml_s_w,cmap="RdBu")
    ax[2].set_title(r'ML Solution of Signal+White Noise ($\alpha = 0$)', fontsize=14)
    ax[1].imshow(ml_s_r,cmap="RdBu")
    ax[1].set_title(f'ML Solution of Signal+Red Noise ($\\alpha = {alpha}$)', fontsize=14)
    ax[0].imshow(signal_map,cmap="RdBu")
    ax[0].set_title("Signal Map", fontsize=14)
    cbar1=fig.colorbar(ax[2].imshow(ml_s_w,cmap="RdBu"), ax=ax[2])
    cbar2=fig.colorbar(ax[1].imshow(ml_s_r,cmap="RdBu"), ax=ax[1])
    cbar3=fig.colorbar(ax[0].imshow(signal_map,cmap="RdBu"), ax=ax[0])

    cbar1.set_label(r'Amplitude', rotation=270, labelpad=20, fontsize=12)
    cbar2.set_label(r'Amplitude', rotation=270, labelpad=20, fontsize=12)
    cbar3.set_label(r'Amplitude', rotation=270, labelpad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{directory}/{output}_simulated_ML_solutions.png')

elif realistic_CMB == 1:
    switch=int(input("Enter 0 to use a CMB power spectrum or 1 to use a map: "))
    if switch==0:

        ang_ext=int(input("Enter the angular extent of the map in degrees: "))
        file=str(input("Enter the name of the CMB power spectrum file: "))
        ell, DlTT = np.loadtxt(file, usecols=(0, 1), unpack=True)
        ClTT = DlTT * 2 * np.pi / (ell*(ell+1.))

        # Creating a map from the power spectrum
        map_size = ang_ext*60.   # map size in arcminutes
        pix_size = map_size/nscan    # pixel size in arcminutes
        N = nscan  # dimension of the map array

        ## make a 2D array of X and Y coordinates
        X,Y = np.meshgrid(np.linspace(-1,1,N),np.linspace(-1,1,N))

        ## define the radial coordinate R of each pixel. R is real-space counterpart of fourier variable k
        R = np.sqrt(X**2. + Y**2.)

        pix_to_rad = (pix_size/60. * np.pi/180.)   # this is the pixel size in radians
        ell_scale_factor = np.pi /pix_to_rad    # here we connect angular size to the multipoles
        ell2d = R * ell_scale_factor     # this is the Fourier analog of the real-vector R in 2D


        ## create 2D power spectrum
        ClTT_expanded = np.zeros(int(ell2d.max())+1)
        ClTT_expanded[0:(ClTT.size)] = ClTT  # fill in the Cls until the max of the ClTT vector
        ClTT2d = ClTT_expanded[ell2d.astype(int)]

        ## make random realization of a Gaussian field and Fourier transform
        random_array_for_T = np.random.normal(0,1,(N,N))
        FT_random_array_for_T = np.fft.fft2(random_array_for_T)   # take FFT since we are in Fourier space 
            
        FT_2d = np.sqrt(ClTT2d) * FT_random_array_for_T # we take the sqrt since the power spectrum is T^2
            
            
        # move back from ell space to real space
        CMB_T = np.fft.ifft2(np.fft.fftshift(FT_2d)) 
        # move back to pixel space for the map
        CMB_T = CMB_T/pix_to_rad
        # we only want to plot the real component
        CMB_T = np.real(CMB_T)

    else:
        map_path=str(input("Enter the file name of the CMB map: "))
        CMB_T=np.load(map_path)

    # Creating the signal from the map
    signal=np.concatenate([CMB_T.reshape(-1), CMB_T.T.reshape(-1)])
    signal_map=CMB_T

    # Simulating the red noise
    r_noise,r_iN=simul_noise(fknee, alpha, nsamp, N_white)
    # Simulating the white noise
    w_noise,w_iN=simul_noise(fknee, 0, nsamp, N_white)

    #plotting the signal, signal map, noise signals and signal + noises
    fig,ax=plt.subplots(3,2,figsize=(15,15))
    ax[0,0].plot(np.arange(len(signal)),signal)
    ax[0,0].set_xlabel("Sample", fontsize=14)
    ax[0,0].set_ylabel(r"Temperature ($\mu K$)", fontsize=14)
    ax[0,0].set_title("Signal", fontsize=14)
    ax[0,1].imshow(signal_map,cmap="RdBu")  
    ax[0,1].set_title("Signal Map", fontsize=14)
    ax[1,0].plot(np.arange(len(r_noise)),r_noise)
    ax[1,0].set_xlabel("Sample", fontsize=14)
    ax[1,0].set_ylabel(r"Temperature ($\mu K$)", fontsize=14)
    ax[1,0].set_title(f"Red Noise ($\\alpha = {alpha}$)", fontsize=14)
    ax[1,1].plot(np.arange(len(w_noise)),w_noise)
    ax[1,1].set_xlabel("Sample", fontsize=14)
    ax[1,1].set_ylabel(r"Temperature ($\mu K$)", fontsize=14)
    ax[1,1].set_title(r'White Noise ($\alpha = 0$)', fontsize=14)
    ax[2,0].plot(np.arange(len(signal)),signal+r_noise)
    ax[2,0].set_xlabel("Sample", fontsize=14)
    ax[2,0].set_ylabel(r"Temperature ($\mu K$)", fontsize=14)
    ax[2,0].set_title(f"Signal+Red Noise ($\\alpha = {alpha}$)", fontsize=14)
    ax[2,1].plot(np.arange(len(signal)),signal+w_noise)
    ax[2,1].set_xlabel("Sample", fontsize=14)
    ax[2,1].set_ylabel(r"Temperature ($\mu K$)", fontsize=14)
    ax[2,1].set_title(r'Signal+White Noise ($\alpha = 0$)', fontsize=14)
    cbar1=fig.colorbar(ax[0,1].imshow(signal_map,cmap="RdBu"), ax=ax[0,1])
    cbar1.set_label(r'Temperature ($\mu K$)', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{directory}/{output}_simulated_signal_noise_REALmaps.png')

    if nside==100:
        nperseg=1024
    elif nside==200:
        nperseg=2048
    else:
        nperseg=4096
    # Calculating the signal and red noise PSD
    psd_signal_red, psd_noise_red, psd_signal_noise_red = calculate_psd(signal, r_noise, nperseg=nperseg)
    # Calculating the signal and white noise PSD
    psd_signal_white, psd_noise_white, psd_signal_noise_white = calculate_psd(signal, w_noise, nperseg=nperseg)
    # Plotting the PSD
    plot_psd(psd_signal_red, psd_noise_red, psd_signal_noise_red, 'Red',modif=True)
    plt.savefig(f'{directory}/{output}_simulated_signal_red_noise_REALpsd.png')
    plot_psd(psd_signal_white, psd_noise_white, psd_signal_noise_white, 'White',modif=True)
    plt.savefig(f'{directory}/{output}_simulated_signal_white_noise_REALpsd.png')

    # ML solutions of the signals
    ml_s_w=mapmaker_ml(signal+w_noise,P_nn,w_iN)
    ml_s_r=mapmaker_ml(signal+r_noise,P_nn,r_iN)

    # Plotting the ML solutions
    fig,ax=plt.subplots(1,3,figsize=(19,5))
    ax[2].imshow(ml_s_w,cmap="RdBu")
    ax[2].set_title(r'ML Solution of Signal+White Noise ($\alpha = 0$)', fontsize=14)
    ax[1].imshow(ml_s_r,cmap="RdBu")
    ax[1].set_title(f'ML Solution of Signal+Red Noise ($\\alpha = {alpha}$)', fontsize=14)
    ax[0].imshow(signal_map,cmap="RdBu")
    ax[0].set_title("Signal Map", fontsize=14)
    cbar1=fig.colorbar(ax[2].imshow(ml_s_w,cmap="RdBu"), ax=ax[2])
    cbar2=fig.colorbar(ax[1].imshow(ml_s_r,cmap="RdBu"), ax=ax[1])
    cbar3=fig.colorbar(ax[0].imshow(signal_map,cmap="RdBu"), ax=ax[0])

    cbar1.set_label(r'Temperature ($\mu K$)', rotation=270, labelpad=20, fontsize=12)
    cbar2.set_label(r'Temperature ($\mu K$)', rotation=270, labelpad=20, fontsize=12)
    cbar3.set_label(r'Temperature ($\mu K$)', rotation=270, labelpad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{directory}/{output}_simulated_ML_solutions_REALmaps.png')

else:
    print("Invalid input for realistic_CMB. Please enter 0 or 1.")


end_time = time.time()
print(f"Time taken to run the code: {end_time-start_time} seconds!!!")