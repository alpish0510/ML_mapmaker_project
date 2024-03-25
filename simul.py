import numpy as np
import scipy
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse.linalg import cg, LinearOperator
import solver as slv
import argparse
import os

directory ='Simulations and Maps'
if not os.path.exists(directory):
    os.makedirs(directory)

parser=argparse.ArgumentParser(description='Simulate a CMB map and produce many types of maps outputs')
parser.add_argument("nside", type=int, help='Number of pixels per side of the output map')
parser.add_argument("output", type=str, help='Name of the output file')
parser.add_argument("fknee", type=float, help='knee frequency of the noise')
parser.add_argument("alpha", type=float, help='spectral index of the noise')
#parser.add_argument("signal analysis", type=int, help='white noise level')
parser.add_argument("--N_w", type=float, default="1.8",help='white noise level')
args=parser.parse_args()

nside = args.nside
output = args.output
fknee = args.fknee
alpha = args.alpha
#signal_analysis = args.signal_analysis
N_white = args.N_w

nscan = nside*4 # Number of samples per row. 
npix  = nside**2
pix_pat1 = (np.mgrid[:nscan,:nscan]*nside/nscan).reshape(2,-1)
pix_pat2 = pix_pat1[::-1] # swap x and y for other pattern
pix      = np.concatenate([pix_pat1,pix_pat2],1)  # Combine the two patterns
nsamp    = pix.shape[1]

# Build a nearest neighbor sparse pointing matrix
iy, ix  = np.floor(pix+0.5).astype(int)%nside  # Nearest pixel
P_nn    = scipy.sparse.csr_array((np.full(nsamp,1),(np.arange(nsamp),iy*nside+ix)),shape=(nsamp,npix))

# Generating noise and signal
fknee = 0.01
freq  = np.fft.rfftfreq(nsamp)
alpha =-3.5
iN    = 1/(N_white*(1+(np.maximum(freq,freq[1]/2)/fknee)**alpha))
iN    = np.maximum(iN, np.max(iN)*1e-8) # Avoid numerical issues

def calc_l(n):
	ly = np.fft.fftfreq(n)[:,None]
	lx = np.fft.fftfreq(n)[None,:]
	l  = (ly**2+lx**2)**0.5
	return l

l      = calc_l(nscan)*nscan/nside # express in units of output pixels
lnorm  = 1
C      = (np.maximum(l,l[0,1]/2)/lnorm)**-2
# Make band-limited by applying a beam. nscan/nside translates from
# the target pixels to the sample spacing
bsigma = 3
B      = np.exp(-0.5*l**2*bsigma**2)

def fmul(f,x):
	if np.asarray(f).size == 1: return f*x
	else: return np.fft.irfft(np.fft.rfft(x)*f,n=len(x)).real

def sim_signal(C, B, nscan):
	signal_map = np.fft.ifft2(np.fft.fft2(np.random.standard_normal((nscan,nscan)))*C**0.5*B).real
	signal = np.concatenate([signal_map.reshape(-1), signal_map.T.reshape(-1)])
	return signal, signal_map

def sim_noise(iN, nsamp):
	noise  = fmul(iN**-0.5, np.random.standard_normal(nsamp))
	return noise

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

def mapmaker_bin(tod, P):
	return scipy.sparse.linalg.spsolve(P.T.dot(P), P.T.dot(tod)).reshape(nside, nside)

def mapmaker_filter_bin(tod, P, F):
	return scipy.sparse.linalg.spsolve(P.T.dot(P), P.T.dot(fmul(F,tod))).reshape(nside, nside)

sig, sig_map = sim_signal(C, B, nscan)
noise        = sim_noise(iN, nsamp)
mapML        = mapmaker_ml(sig,P_nn,iN)
noiseML      = mapmaker_ml(noise,P_nn,iN)
mapBin       = mapmaker_bin(sig,P_nn)
noiseBin     = mapmaker_bin(noise,P_nn)

# Plotting the results
fig,ax=plt.subplots(3,2,figsize=(15,17))
ax[0,0].plot(np.arange(len(sig)),sig)
ax[0,0].set_xlabel("Sample", fontsize=14)
ax[0,0].set_ylabel("Temperature ($\mu K$)", fontsize=14)
ax[0,0].set_title("Signal", fontsize=14)
ax[0,1].imshow(sig_map,cmap="RdBu")
ax[0,1].set_title("Signal Map", fontsize=14)
ax[1,0].imshow(mapML,cmap="RdBu")
ax[1,0].set_title("Map ML", fontsize=14)
ax[1,1].imshow(noiseML,cmap="RdBu")
ax[1,1].set_title("Noise ML", fontsize=14)
ax[2,0].imshow(mapBin,cmap="RdBu")
ax[2,0].set_title("Map Binned", fontsize=14)
ax[2,1].imshow(noiseBin,cmap="RdBu")
ax[2,1].set_title("Noise Binned", fontsize=14)
cbar1=fig.colorbar(ax[0,1].imshow(sig_map,cmap="RdBu"), ax=ax[0,1])
cbar2=fig.colorbar(ax[1,0].imshow(mapML,cmap="RdBu"), ax=ax[1,0])
cbar3=fig.colorbar(ax[1,1].imshow(noiseML,cmap="RdBu"), ax=ax[1,1])
cbar4=fig.colorbar(ax[2,0].imshow(mapBin,cmap="RdBu"), ax=ax[2,0])
cbar5=fig.colorbar(ax[2,1].imshow(noiseBin,cmap="RdBu"), ax=ax[2,1])
# adding label on the colorbars
cbar1.set_label(r'Temperature ($\mu K$)', rotation=270, labelpad=20, fontsize=12)
cbar2.set_label(r'Temperature ($\mu K$)', rotation=270, labelpad=20, fontsize=12)
cbar3.set_label(r'Temperature ($\mu K$)', rotation=270, labelpad=20, fontsize=12)
cbar4.set_label(r'Temperature ($\mu K$)', rotation=270, labelpad=20, fontsize=12)
cbar5.set_label(r'Temperature ($\mu K$)', rotation=270, labelpad=20, fontsize=12)
# Adding title to the plot
fig.suptitle("Simulated signal and noise maps with ML and Binned Solutions", fontsize=16, y=0.99)

# Saving the results
plt.tight_layout()
plt.savefig(directory+'/'+output+'SigNaess.png')
