import numpy as np
import matplotlib.pyplot as plt
import pysm3
import pysm3.units as u
import healpy as hp
import os
import solver as slv
import argparse
import matplotlib.gridspec as gridspec
from scipy.sparse.linalg import cg, LinearOperator
import time

start_time = time.time()
directory="Maps"
if not os.path.exists(directory):
    os.makedirs(directory)

parser=argparse.ArgumentParser(description='Simulate a CMB map and produce many types of maps outputs')
parser.add_argument("nside", type=int, help='The healpix nside of the output map')
parser.add_argument("freq", type=float, help='The frequency of the map in GHz')
parser.add_argument("angular_size", type=int, help='Angular size of the map in degrees')
parser.add_argument("out_map_size", type=int, help='Output map size in pixels')
parser.add_argument("--mr", type=bool, default=False, help='If True, the mean of the map will be removed')
parser.add_argument("--sm", type=bool, default=False, help='Smoothing the map with a Gaussian beam, if True, the user will be asked to enter the scale of the smoothing or FWHM of the beam in degrees.')

args=parser.parse_args()
nside = args.nside
freq = args.freq
angular_size = args.angular_size
out_map_size = args.out_map_size
mean_rem = args.mr
smoothing = args.sm

lonra=[-angular_size/2,angular_size/2]
latra=[-angular_size/2,angular_size/2]

# Generate a CMB map    
sky = pysm3.Sky(nside=4096, preset_strings=["c4"])
map_allsky = sky.get_emission(freq * u.GHz)
map_allsky = map_allsky.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(220*u.GHz))
map=hp.visufunc.cartview(map_allsky[0], unit=map_allsky.unit, cmap='RdBu',lonra=lonra,latra=latra,remove_dip=True,return_projected_map=True,xsize=out_map_size)
map=np.array(map)
if mean_rem==True:
    if smoothing == True:
        scale=input("Enter the scale of the smoothing or FWHM of the beam in degrees: ")
        scale=float(scale)*np.pi/180
        iden=scale*180/np.pi
        smoo_map=hp.sphtfunc.smoothing(map_allsky[0], sigma=scale,lmax=5000)
        smoo_map=hp.visufunc.cartview(smoo_map, unit=map_allsky.unit, cmap='RdBu',lonra=lonra,latra=latra,remove_dip=True,return_projected_map=True,xsize=out_map_size)
        final_map=smoo_map-np.mean(smoo_map)
        final_map=np.array(final_map)
        plt.figure(figsize=(5,5))
        plt.imshow(final_map,cmap='RdBu')
        cbar=plt.colorbar()
        plt.title(f"CMB map at {freq} GHz")
        cbar.set_label('Temperature (uK)', rotation=270, labelpad=20, fontsize=12)
        plt.tight_layout()
        plt.savefig(directory+f"/smoo{iden}_pysm3_{out_map_size}_res{nside}MEANREM.png")
        #Saving the map
        np.save(directory+f"/smoo{iden}_pysm3_{out_map_size}_res{nside}MEANREM",final_map)
    else:
        final_map=map-np.mean(map)
        final_map=np.array(final_map)
        plt.figure(figsize=(5,5))
        plt.imshow(final_map,cmap='RdBu')
        cbar=plt.colorbar()
        plt.title(f"CMB map at {freq} GHz")
        cbar.set_label('Temperature (uK)', rotation=270, labelpad=20, fontsize=12)
        plt.tight_layout()
        plt.savefig(directory+f"/pysm3_{out_map_size}_res{nside}MEANREM.png")
        #Saving the map
        np.save(directory+f"/pysm3_{out_map_size}_res{nside}MEANREM",final_map)
else:
    final_map=map
    if smoothing == True:
        scale=input("Enter the scale of the smoothing or FWHM of the beam in degrees: ")
        scale=float(scale)*np.pi/180
        iden=scale*180/np.pi
        smoo_map=hp.sphtfunc.smoothing(map_allsky[0], sigma=scale,lmax=5000)
        smoo_map=hp.visufunc.cartview(smoo_map, unit=map_allsky.unit, cmap='RdBu',lonra=lonra,latra=latra,remove_dip=True,return_projected_map=True,xsize=out_map_size)
        final_map=np.array(smoo_map)
        plt.figure(figsize=(5,5))
        plt.imshow(final_map,cmap='RdBu')
        cbar=plt.colorbar()
        plt.title(f"CMB map at {freq} GHz")
        cbar.set_label('Temperature (uK)', rotation=270, labelpad=20, fontsize=12)
        plt.tight_layout()
        plt.savefig(directory+f"/smoo{iden}_pysm3_{out_map_size}_res{nside}.png")
        #Saving the map
        np.save(directory+f"/smoo{iden}_pysm3_{out_map_size}_res{nside}",final_map)

    else:
        plt.figure(figsize=(5,5))
        plt.imshow(final_map,cmap='RdBu')
        cbar=plt.colorbar()
        plt.title(f"CMB map at {freq} GHz")
        cbar.set_label('Temperature (uK)', rotation=270, labelpad=20, fontsize=12)
        plt.tight_layout()
        plt.savefig(directory+f"/pysm3_{out_map_size}_res{nside}.png")

        #Saving the map
        np.save(directory+f"/pysm3_{out_map_size}_res{nside}",final_map)

end_time = time.time()
print(f"Time taken to run the code: {end_time-start_time} seconds!!!")
