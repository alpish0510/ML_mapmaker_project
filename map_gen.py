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

# Start time of the code
start_time = time.time()

# Setting up the directory 
directory="Maps"
if not os.path.exists(directory):
    os.makedirs(directory)

# Setting up the arguments
parser=argparse.ArgumentParser(description='Simulate a CMB map and produce many types of maps outputs')
parser.add_argument("nside", type=int, help='The healpix nside of the output map. (Ex: 512, 1024, 2048, 4096)')
parser.add_argument("freq", type=float, help='The frequency of the map in GHz (Ex: 200)')
parser.add_argument("angular_size", type=int, help='Angular size of the map in degrees. (Ex: 4)')
parser.add_argument("out_map_size", type=int, help='Output map size in pixels. (Ex: 100, 200, etc.)')
parser.add_argument("RA", type=float, help='Right Ascension of the source in degrees')
parser.add_argument("DEC", type=float, help='Declination of the source in degrees')
parser.add_argument("--sm", type=bool, default=False, help='Smoothing the map with a Gaussian beam, if True, the user will be asked to enter the scale of the smoothing or FWHM of the beam in degrees.')
parser.add_argument("--model", type=str, default='c4', help='The model of the sky to be used')

# Parsing the arguments
args=parser.parse_args()
nside = args.nside
freq = args.freq
angular_size = args.angular_size
out_map_size = args.out_map_size
RA = args.RA
DEC = args.DEC
smoothing = args.sm
model = args.model

# Conditional statements to setup mutiple models. Documentation at https://pysm3.readthedocs.io/en/latest/models.html
if len(model) == 2:
    m1=model
    m2=None
    m3=None
    sky = pysm3.Sky(nside=nside, preset_strings=[f"{m1}"])
elif len(model) == 5:
    m1=model[:2]
    m2=model[3:5]
    m3=None
    sky = pysm3.Sky(nside=nside, preset_strings=[f"{m1}",f"{m2}"])
elif len(model)== 8:
    m1=model[:2]
    m2=model[3:5]
    m3=model[6:8]
    sky = pysm3.Sky(nside=nside, preset_strings=[f"{m1}",f"{m2}",f"{m3}"])

# extent of the map for hp.visufunc.cartview()
lonra=[-angular_size/2,angular_size/2]
latra=[-angular_size/2,angular_size/2]

# Generating the map
map_allsky = sky.get_emission(freq * u.GHz)
map_allsky = map_allsky.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz))

if smoothing == True:
    scale=input("Enter the scale of the smoothing or FWHM of the beam in degrees: ")
    map_projected=pysm3.apply_smoothing_and_coord_transform(map_allsky[0],fwhm=scale*u.deg,rot=hp.Rotator(coord=("G","C")),map2alm_lsq_maxiter=1000)
    map=hp.visufunc.cartview(map_projected, unit=map_allsky.unit, cmap='RdBu',rot=[RA,DEC],lonra=lonra,latra=latra,remove_dip=True,return_projected_map=True,xsize=out_map_size)
    map=np.array(map)
    smoo_map=np.flip(map,axis=0)
    final_map=smoo_map
    plt.figure(figsize=(5,5))
    plt.imshow(final_map,cmap='RdBu')
    cbar=plt.colorbar()
    plt.title(f"CMB map at {freq} GHz")
    cbar.set_label('Temperature (uK)', rotation=270, labelpad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(directory+f"/smoo{scale}_pysm3_{out_map_size}_res{nside}.png")
    #Saving the map
    np.save(directory+f"/smoo{scale}_pysm3_{out_map_size}_res{nside}",final_map)
else:
    map_projected=pysm3.apply_smoothing_and_coord_transform(map_allsky[0],rot=hp.Rotator(coord=("G","C")),map2alm_lsq_maxiter=1000)
    map=hp.visufunc.cartview(map_projected, unit=map_allsky.unit, cmap='RdBu',rot=[RA,DEC],lonra=lonra,latra=latra,remove_dip=True,return_projected_map=True,xsize=out_map_size)
    map=np.array(map)
    final_map=np.flip(map,axis=0)
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
