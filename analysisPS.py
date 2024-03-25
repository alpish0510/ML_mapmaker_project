import numpy as np
import scipy
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse.linalg import cg, LinearOperator
import solver as slv
import argparse
import os
import time

start_time = time.time()

directory = 'Output'

if not os.path.exists(directory):
    os.makedirs(directory)


# Create the parser
parser = argparse.ArgumentParser(description="take in a pysm3 map and solve for a power spectrum")
parser.add_argument("map",nargs='+',type=str, help="the map to solve for the power spectrum")
parser.add_argument("ang_ext", help="the angular extent (in degrees) of the input map")
parser.add_argument("binning", help="binning of the power spectrum")
parser.add_argument("noise", help="Do you want to add noise to the map? 1 for yes, 0 for no.")
parser.add_argument("--N_w", default=1.8, help="White noise level (in the unit of uK^2)")
parser.add_argument("--fknee", default=0.01, help="Knee frequency (in the unit of Hz)")

# Parse the arguments
args = parser.parse_args()
markers=[np.array(map_name[:-4]) for map_name in args.map]
maps = [np.array(np.load(map_path)) for map_path in args.map]

# Load the arguments to variables
if len(args.map) == 1:
    map = maps[0]
    marker= markers[0]
elif len(args.map) == 2:
    map1 = maps[0]
    map2 = maps[1]
    marker1= markers[0]
    marker2= markers[1]
else:
    map1 = maps[0]
    map2 = maps[1]
    map3 = maps[2]
    marker1= markers[0]
    marker2= markers[1]
    marker3= markers[2]


ang_ext=float(args.ang_ext)
binning=int(args.binning)
noise=int(args.noise)
N_w=float(args.N_w)
fknee=float(args.fknee)

bin_size=binning

if len(args.map) == 1:
    #print(f"{marker}")
    print("We have a single input map!")
    print(f"The angular extent of the input map is {ang_ext} degrees")
    
    # Power spectrum of the pysm3 maps
    ellm,dlm=slv.ps_calc(map,ang_ext,delta_ell=bin_size)
    print("Calculating the power spectrum of the input map...")
    print(f"The binning of the power spectrum is $\Delta \ell$ = {binning}")
    # Create a figure
    fig = plt.figure(figsize=(18, 6))

    # Create the first subplot for the power spectrum
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.semilogy(ellm, dlm, 'r', label=f'map shape {map.shape[0]}x{map.shape[1]}')
    ax1.set_xlabel('$\ell$', fontsize=17)
    ax1.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(fontsize=13)
    ax1.set_title(f'Power spectrum of the {marker} map', fontsize=17)
    
    # Create the second subplot for the map
    ax2 = fig.add_subplot(1, 2, 1)
    im = ax2.imshow(map, cmap='RdBu', interpolation='bilinear')
    cbar = fig.colorbar(im, ax=ax2, orientation='vertical')
    cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
    ax2.set_title(f'{marker} map', fontsize=17)
    # Adjust the layout and save the figure

    plt.savefig(f'{directory}/{marker}_map_PS.png')
    plt.show()
    print(f"Power spectrum of the input map has been calculated successfully! ----> {marker}_map_PS.png")
    # Create the MLsolver object
    nside=int(map.shape[0]/4)
    mlsolver = slv.MLsolver(map, nside, N_w, fknee)
    print("Creating the MLsolver object...")
    if noise==0:
        print("You chose not to add noise to the input map!")
        sol=mlsolver.ml_solver(0,add_noise=False)  #white noise solution
        ell,dl=slv.ps_calc(sol,ang_ext,delta_ell=bin_size)

        # Create a figure
        fig = plt.figure(figsize=(18, 5))
        # Create the first subplot for the power spectrum
        ax1 = fig.add_subplot(1, 2, 2)
        ax1.semilogy(ell, dl, 'r', label=f'ML solution {sol.shape[0]}x{sol.shape[1]}')
        ax1.set_xlabel('$\ell$', fontsize=17)
        ax1.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.legend(fontsize=13)
        ax1.set_title('Power spectrum of the ML map', fontsize=17)

        # Create the second subplot for the map
        ax2 = fig.add_subplot(1, 2, 1)
        im = ax2.imshow(sol, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax2, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax2.set_title('ML solution', fontsize=17)
        # Adjust the layout and save the figure
        plt.savefig(f'{directory}/{marker}_Solution_without_noise.png')
        plt.show()
        print(f"The ML solution and its power spectrum has been calculated successfully! ---->  {marker}_Solution_without_noise.png")
        # plotting the comparision power spectrum
        plt.figure(figsize=(18,7))
        plt.semilogy(ellm,dlm,label=f'{marker} map {map.shape[0]}x{map.shape[1]}')
        plt.semilogy(ell,dl,label=f'ML solution {sol.shape[0]}x{sol.shape[1]}')
        plt.xlabel('$\ell$', fontsize=17)
        plt.ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize=13)
        plt.title('Power spectrum of the input map and the ML solution', fontsize=17)
        plt.savefig(f'{directory}/{marker}_ComparisionPS.png')
        plt.show()
        print(f"The comparision power spectrum has been calculated successfully! ---->  {marker}_ComparisionPS.png")
        print("The residual power spectrum is being calculated...")
        # normalizing the residual PS
        norm=dl/np.mean(dl)
        normm=dlm/np.mean(dlm)
        
        # calculating the residual PS
        res=(norm-normm)/normm * 100

        
        
        #plotting the residual PS
        plt.figure(figsize=(18,7))
        # Create a figure
        fig = plt.figure(figsize=(18,10))

    # Define the grid
        gs = gridspec.GridSpec(2, 1)

    # Create the first subplot
        ax1 = fig.add_subplot(gs[:1, :])
        ax1.tick_params(axis='y', labelsize=14)  # Change the size of the y-axis ticks
        ax1.tick_params(axis='x', labelsize=14)  # Change the size of the x-axis ticks
        ax1.semilogy(ell,norm,label=f'{nside}x{nside} ML map',c='orange')
        ax1.semilogy(ellm,normm,label=f'{map.shape[0]}x{map.shape[1]} {marker} map',ls='-.',c='black')
        ax1.set_xlabel('$\ell$', fontsize=17)
        ax1.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        ax1.legend(fontsize=13)
        ax1.set_title(f'Normalized power spectrum of the {marker} map and the ML solution', fontsize=17)
        
    # Create the second subplot
        ax2 = fig.add_subplot(gs[1, :])
        ax2.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.scatter(ell,res,label=f'{map.shape[0]}-->{nside} Percentage residuals',c='orange')
        ax2.axhline(0, color='r', lw=1, ls='--')
        ax2.set_xlabel('$\ell$',fontsize=14)
        ax2.set_ylabel('Percentage residuals',fontsize=14)
        ax2.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
        plt.savefig(f'{directory}/{marker}_ResidualPS.png')
        plt.show()
        print(f"The residual power spectrum has been calculated successfully! ----> {marker}_ResidualPS.png")

    elif noise==1:
        print("You chose to add noise to the input map!")

        sol=mlsolver.ml_solver(0,add_noise=True)

        solw=mlsolver.ml_solver(0,add_noise=True)  #white noise solution
        solr=mlsolver.ml_solver(-3.5,add_noise=True)  #red noise solution
        ellw,dlw=slv.ps_calc(solw,ang_ext,delta_ell=bin_size)
        ellr,dlr=slv.ps_calc(solr,ang_ext,delta_ell=bin_size)
        
        # Create a figure
        fig = plt.figure(figsize=(14, 10))

        # Define the grid
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

        # Create the first subplot for the map
        ax1 = plt.subplot(gs[0, 0])
        im = ax1.imshow(solw, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax1, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax1.set_title('ML solution (W.N.)', fontsize=17)

        # Create the second subplot for the map
        ax2 = plt.subplot(gs[0, 1])
        im = ax2.imshow(solr, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax2, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax2.set_title('ML solution (R.N.)', fontsize=17)

        # Create the third subplot for the power spectrum
        ax3 = plt.subplot(gs[1, :])
        ax3.semilogy(ellw, dlw, 'r', label=f'ML solution {solw.shape[0]}x{solw.shape[1]} (W.N.)')
        ax3.semilogy(ellr, dlr, 'b', label=f'ML solution {solr.shape[0]}x{solr.shape[1]} (R.N.)')
        ax3.set_xlabel('$\ell$', fontsize=17)
        ax3.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        ax3.legend(fontsize=13)
        ax3.set_title('Power spectrum of the ML maps', fontsize=17)

        # Adjust the layout and save the figure
        plt.tight_layout()
        plt.savefig(f'{directory}/{marker}_Solutions.png')
        plt.show()
        print(f"The ML solutions and their power spectrum have been calculated successfully! ----> {marker}_Solutions.png")

        # plotting the comparision power spectrum
        plt.figure(figsize=(18,7))
        plt.semilogy(ellm,dlm,label=f'{marker} map {map.shape[0]}x{map.shape[1]}')
        plt.semilogy(ellw,dlw,label=f'ML solution {solw.shape[0]}x{solw.shape[1]} (W.N.)',c='r')
        plt.semilogy(ellr,dlr,label=f'ML solution {solr.shape[0]}x{solr.shape[1]} (R.N.)',c='b')
        ax = plt.gca()  # Get the current axes
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)  # Change the border thickness here
        plt.xlabel('$\ell$', fontsize=17)
        plt.ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize=13)
        plt.title(f'Power spectrum of the {marker} map and the ML solutions', fontsize=17)
        plt.savefig(f'{directory}/{marker}_ComparisionPS.png')
        plt.show()
        print(f"The comparision power spectrum has been calculated successfully! ----> {marker}_ComparisionPS.png")
        print("The residual power spectrum is being calculated...")

        # normalizing the residual PS
        normw=dlw/np.mean(dlw)
        normr=dlr/np.mean(dlr)
        normm=dlm/np.mean(dlm)

        # calculating the residual PS
        resw=(normw-normm)/normm * 100
        resr=(normr-normm)/normm * 100

        #plotting the residual PS
        plt.figure(figsize=(18,7))
        # Create a figure
        fig = plt.figure(figsize=(18,10))

    # Define the grid
        gs = gridspec.GridSpec(3, 1)

    # Create the first subplot
        ax1 = fig.add_subplot(gs[:2, :])
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(3)
        ax1.tick_params(axis='both', labelsize=14, width=2)  # Change the size and weight of the ticks and labels    
        ax1.tick_params(axis='y', labelsize=14)  # Change the size of the y-axis ticks
        ax1.tick_params(axis='x', labelsize=14)  # Change the size of the x-axis ticks
        ax1.semilogy(ellw,normw,label=f'{nside}x{nside} ML map (w.n. model)',c='orange')
        ax1.semilogy(ellm,normm,label=f'{map.shape[0]}x{map.shape[1]} {marker} map',ls='-.',c='black')
        ax1.semilogy(ellr,normr,label=f'{nside}x{nside} ML map (r.n. model)',c='red')
        ax1.set_xlabel('$\ell$', fontsize=17)
        ax1.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        ax1.legend(fontsize=13)
        ax1.set_title(f'Normalized power spectrum of the {marker} map and the ML solutions', fontsize=17)

    # Create the second subplot
        ax2 = fig.add_subplot(gs[2, :])
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(3)
        ax2.tick_params(axis='both', labelsize=14, width=2)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.scatter(ellw,resw,label=f'{map.shape[0]}-->{nside} Percentage residuals white noise',c='orange')
        ax2.scatter(ellr,resr,label=f'{map.shape[0]}-->{nside} Percentage residuals red noise',c='red')
        ax2.axhline(0, color='r', lw=1, ls='--')
        ax2.set_xlabel('$\ell$',fontsize=14)
        ax2.set_ylabel('Percentage residuals',fontsize=14)
        ax2.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        
        plt.tight_layout()
        plt.savefig(f'{directory}/{marker}_ResidualPS.png')

        print(f"The residual power spectrum has been calculated successfully! ----> {marker}_ResidualPS.png")

    else:
        raise ValueError("The noise parameter should be 0 or 1")
    
    

elif len(args.map) == 2:
    #print(f"{marker1} and {marker2}")
    print("we have 2 maps!")
    print(f"The angular extent of the input maps is {ang_ext} degrees")
    # Power spectrum of the pysm3 maps
    ellm,dlm=slv.ps_calc(map1,ang_ext,delta_ell=bin_size)
    ellm2,dlm2=slv.ps_calc(map2,ang_ext,delta_ell=bin_size)
    print(f"Calculating the power spectrum of the {marker1} and {marker2} maps...")
    print(f"The binning of the power spectrum is $\Delta \ell$ = {binning}")
   
    # Create a figure
    fig = plt.figure(figsize=(18, 10))
    # Create a gridspec
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,1])

    # Create the first subplot for the power spectrum
    ax1 = fig.add_subplot(gs[1, :])
    # making the spines thicker
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(3)
    ax1.semilogy(ellm, dlm, 'r', label=f'{marker1} map {map1.shape[0]}x{map1.shape[1]}')
    ax1.semilogy(ellm2, dlm2, 'b', label=f'{marker2} map {map2.shape[0]}x{map2.shape[1]}')
    ax1.set_xlabel('$\ell$', fontsize=17)
    ax1.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
    # making the ticks thicker
    ax1.tick_params(axis='both', which='major', labelsize=14, width=2)
    ax1.legend(fontsize=13)
    ax1.set_title(f'Power spectrum of the {marker1} and {marker2} maps', fontsize=17)

    # Create the second subplot for the map
    ax2 = fig.add_subplot(gs[0, 0])
    im = ax2.imshow(map1, cmap='RdBu', interpolation='bilinear')
    cbar = fig.colorbar(im, ax=ax2, orientation='vertical')
    cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
    ax2.set_title(f'{marker1} map', fontsize=17)

    # Create the third subplot for the map
    ax3 = fig.add_subplot(gs[0, 1])
    im = ax3.imshow(map2, cmap='RdBu', interpolation='bilinear')
    cbar = fig.colorbar(im, ax=ax3, orientation='vertical')
    cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
    ax3.set_title(f'{marker2} map', fontsize=17)

    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig(f'{directory}/{marker1}_{marker2}_map_PS.png')
    plt.show()
    print(f"Power spectrum of the input maps has been calculated successfully! ----> {marker1}_{marker2}_Input_map_PS.png")
    # Create the MLsolver object
    nside1=int(map1.shape[0]/4)
    nside2=int(map2.shape[0]/4)
    mlsolver = slv.MLsolver(map1, nside1)
    mlsolver2 = slv.MLsolver(map2, nside2)
    print("Creating the MLsolver objects...")
    if noise==0:
        print("You chose not to add noise to the input maps!")
        sol=mlsolver.ml_solver(0,add_noise=False)
        sol2=mlsolver2.ml_solver(0,add_noise=False)
        ell,dl=slv.ps_calc(sol,ang_ext,delta_ell=bin_size)
        ell2,dl2=slv.ps_calc(sol2,ang_ext,delta_ell=bin_size)
        # Create a figure
        fig = plt.figure(figsize=(18, 10))
        # Create a gridspec
        gs = gridspec.GridSpec(2, 2, height_ratios=[1,1])

        # Create the first subplot for the power spectrum
        ax1 = fig.add_subplot(gs[1, :])
        # making the spines thicker
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(3)
        ax1.semilogy(ell, dl, 'r', label=f'ML solution {sol.shape[0]}x{sol.shape[1]}')
        ax1.semilogy(ell2, dl2, 'b', label=f'ML solution {sol2.shape[0]}x{sol2.shape[1]}')
        ax1.set_xlabel('$\ell$', fontsize=17)
        ax1.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        # making the ticks thicker
        ax1.tick_params(axis='both', which='major', labelsize=14, width=2)
        ax1.legend(fontsize=13)
        ax1.set_title('Power spectrum of the ML maps', fontsize=17)

        # Create the second subplot for the map
        ax2 = fig.add_subplot(gs[0, 0])
        im = ax2.imshow(sol, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax2, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax2.set_title('ML solution 1', fontsize=17)
        
        # Create the third subplot for the map
        ax3 = fig.add_subplot(gs[0, 1])
        im = ax3.imshow(sol2, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax3, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax3.set_title('ML solution 2', fontsize=17)

        # Adjust the layout and save the figure
        plt.tight_layout()
        plt.savefig(f'{directory}/{marker1}_{marker2}_Solution_without_noise.png')
        plt.show()
        print(f"The ML solutions and their power spectrum have been calculated successfully! ----> {marker1}_{marker2}_Solution_without_noise.png")

        # plotting the comparision power spectrum
        plt.figure(figsize=(18,7))
        plt.semilogy(ellm,dlm,label=f'{marker1} map {map1.shape[0]}x{map1.shape[1]}',ls='-.',c='black')
        plt.semilogy(ell,dl,label=f'ML solution 1 {sol.shape[0]}x{sol.shape[1]}',c='black')
        plt.semilogy(ellm2,dlm2,label=f'{marker2} map {map2.shape[0]}x{map2.shape[1]}',ls='-.',c='blue')
        plt.semilogy(ell2,dl2,label=f'ML solution 2 {sol2.shape[0]}x{sol2.shape[1]}',c='blue')
        ax = plt.gca()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        plt.xlabel('$\ell$', fontsize=17)
        plt.ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize=13)
        plt.title('Power spectrum of the input maps and the ML solutions', fontsize=17)
        plt.savefig(f'{directory}/{marker1}_{marker2}_ComparisionPS.png')
        plt.show()
        print(f"The comparision power spectrum has been calculated successfully! ----> {marker1}_{marker2}_ComparisionPS.png")
        print("The residual power spectrum is being calculated...")
        # normalizing the residual PS
        norm=dl/np.mean(dl)
        normm=dlm/np.mean(dlm)
        norm2=dl2/np.mean(dl2)
        normm2=dlm2/np.mean(dlm2)
        # calculating the residual PS
        res=(norm-normm)/normm * 100
        res2=(norm2-normm2)/normm2 * 100
        #plotting the residual PS
        plt.figure(figsize=(18,7))
        # Create a figure
        fig = plt.figure(figsize=(18,10))

    # Define the grid
        gs = gridspec.GridSpec(3, 1)
        
    # Create the first subplot
        ax1 = fig.add_subplot(gs[:2, :])
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(3)
        ax1.tick_params(axis='both', labelsize=14, width=2)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.plot(ell,norm,label=f'{nside1}x{nside1} ML map 1',c='orange')
        ax1.plot(ellm,normm,label=f'{map1.shape[0]}x{map1.shape[1]} {marker1} map',ls='-.',c='black')
        ax1.plot(ell2,norm2+5,label=f'{nside2}x{nside2} ML map 2',c='red')
        ax1.plot(ellm2,normm2+5,label=f'{map2.shape[0]}x{map2.shape[1]} {marker1} map',ls='-.',c='blue')
        ax1.set_xlabel('$\ell$', fontsize=17)
        ax1.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        ax1.legend(fontsize=13)
        ax1.set_title(f'Normalized power spectrum of the {marker1} and {marker2} maps and the ML solutions', fontsize=17)

    # Create the second subplot
        ax2 = fig.add_subplot(gs[2, :])
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(3)
        ax2.tick_params(axis='both', labelsize=14, width=2)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.scatter(ell,res,label=f'{map1.shape[0]}-->{nside1} Percentage residuals',c='orange')
        ax2.scatter(ell2,res2,label=f'{map2.shape[0]}-->{nside2} Percentage residuals',c='red')
        ax2.axhline(0, color='r', lw=1, ls='--')
        ax2.set_xlabel('$\ell$',fontsize=14)
        ax2.set_ylabel('Percentage residuals',fontsize=14)
        ax2.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        plt.tight_layout()
        plt.savefig(f'{directory}/{marker1}_{marker2}_ResidualPS.png')
        plt.show()
        print(f"The residual power spectrum has been calculated successfully! ----> {marker1}_{marker2}_ResidualPS.png")

    elif noise==1:
        print("You chose to add noise to the input maps!")
        solw=mlsolver.ml_solver(0,add_noise=True)
        solr=mlsolver.ml_solver(-3.5,add_noise=True)
        solw2=mlsolver2.ml_solver(0,add_noise=True)
        solr2=mlsolver2.ml_solver(-3.5,add_noise=True)
        ellw,dlw=slv.ps_calc(solw,ang_ext,delta_ell=bin_size)
        ellr,dlr=slv.ps_calc(solr,ang_ext,delta_ell=bin_size)
        ellw2,dlw2=slv.ps_calc(solw2,ang_ext,delta_ell=bin_size)
        ellr2,dlr2=slv.ps_calc(solr2,ang_ext,delta_ell=bin_size)
       
        # Create a figure
        fig = plt.figure(figsize=(12, 16))

        # Define the grid
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

        # Create the first subplot for the map
        ax1 = plt.subplot(gs[0, 0])
        im = ax1.imshow(solw, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax1, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax1.set_title('ML solution 1 (W.N.)', fontsize=17)

        # Create the second subplot for the map
        ax2 = plt.subplot(gs[0, 1])
        im = ax2.imshow(solr, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax2, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax2.set_title('ML solution 1 (R.N.)', fontsize=17)

        # Create the third subplot for the map
        ax3 = plt.subplot(gs[1, 0])
        im = ax3.imshow(solw2, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax3, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax3.set_title('ML solution 2 (W.N.)', fontsize=17)

        # Create the fourth subplot for the map
        ax4 = plt.subplot(gs[1, 1])
        im = ax4.imshow(solr2, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax4, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax4.set_title('ML solution 2 (R.N.)', fontsize=17)

        # Create the fifth subplot for the power spectrum
        ax5 = plt.subplot(gs[2, :])
        ax5.semilogy(ellw, dlw, 'r', label=f'ML solution 1 {solw.shape[0]}x{solw.shape[1]} (W.N.)')
        ax5.semilogy(ellr, dlr, 'b', label=f'ML solution 1 {solr.shape[0]}x{solr.shape[1]} (R.N.)')
        ax5.semilogy(ellw2, dlw2, 'g', label=f'ML solution 2 {solw2.shape[0]}x{solw2.shape[1]} (W.N.)')
        ax5.semilogy(ellr2, dlr2, 'y', label=f'ML solution 2 {solr2.shape[0]}x{solr2.shape[1]} (R.N.)')
        ax5.set_xlabel('$\ell$', fontsize=17)
        ax5.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        ax5.tick_params(axis='both', which='major', labelsize=14)
        ax5.legend(fontsize=13)
        ax5.set_title('Power spectrum of the ML maps', fontsize=17)

        # Adjust the layout and save the figure
        plt.tight_layout()
        plt.savefig(f'{directory}/{marker1}_{marker2}_Solutions.png')
        plt.show()
        print(f"The ML solutions and their power spectrum have been calculated successfully! ----> {marker1}_{marker2}_Solutions.png")

        # plotting the comparision power spectrum
        plt.figure(figsize=(18,7))
        plt.semilogy(ellm,dlm,label=f'{marker1} map {map1.shape[0]}x{map1.shape[1]}',ls='-.',c='black')
        plt.semilogy(ellw,dlw,label=f'ML solution 1 {solw.shape[0]}x{solw.shape[1]} (W.N.)',c='r')
        plt.semilogy(ellr,dlr,label=f'ML solution 1 {solr.shape[0]}x{solr.shape[1]} (R.N.)',c='b')
        plt.semilogy(ellm2,dlm2,label=f'{marker2} map {map2.shape[0]}x{map2.shape[1]}',ls='-.',c='blue')
        plt.semilogy(ellw2,dlw2,label=f'ML solution 2 {solw2.shape[0]}x{solw2.shape[1]} (W.N.)',c='g')
        plt.semilogy(ellr2,dlr2,label=f'ML solution 2 {solr2.shape[0]}x{solr2.shape[1]} (R.N.)',c='y')
        ax = plt.gca()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        plt.xlabel('$\ell$', fontsize=17)
        plt.ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize=13)
        plt.title(f'Power spectrum of the {marker1} and {marker2} maps and the ML solutions', fontsize=17)
        plt.savefig(f'{directory}/{marker1}_{marker2}_ComparisionPS.png')
        plt.show()
        print(f"The comparision power spectrum has been calculated successfully! ----> {marker1}_{marker2}_ComparisionPS.png")
        print("The residual power spectrum is being calculated...")
        # normalizing the residual PS
        normw=dlw/np.mean(dlw)
        normr=dlr/np.mean(dlr)
        normm=dlm/np.mean(dlm)
        normw2=dlw2/np.mean(dlw2)
        normr2=dlr2/np.mean(dlr2)
        normm2=dlm2/np.mean(dlm2)
        # calculating the residual PS
        resw=(normw-normm)/normm * 100
        resr=(normr-normm)/normm * 100
        resw2=(normw2-normm2)/normm2 * 100
        resr2=(normr2-normm2)/normm2 * 100
        #plotting the residual PS
        plt.figure(figsize=(18,7))
        # Create a figure
        fig = plt.figure(figsize=(18,10))

    # Define the grid
        gs = gridspec.GridSpec(3, 1)

    # Create the first subplot
        
        ax1 = fig.add_subplot(gs[:2, :])
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(3)
        ax1.tick_params(axis='both', labelsize=14, width=2)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.plot(ellw,normw,label=f'{nside1}x{nside1} ML map 1 (w.n. model)',c='orange')
        ax1.plot(ellm,normm,label=f'{map1.shape[0]}x{map1.shape[1]} {marker1} map',ls='-.',c='orange')
        ax1.plot(ellr,normr,label=f'{nside1}x{nside1} ML map 1 (r.n. model)',c='orange',ls='--')
        ax1.plot(ellw2,normw2+5,label=f'{nside2}x{nside2} ML map 2 (w.n. model)',c='blue')
        ax1.plot(ellm2,normm2+5,label=f'{map2.shape[0]}x{map2.shape[1]} {marker2} map',ls='-.',c='blue')
        ax1.plot(ellr2,normr2+5,label=f'{nside2}x{nside2} ML map 2 (r.n. model)',c='blue',ls='--')
        ax1.set_xlabel('$\ell$', fontsize=17)
        ax1.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        ax1.legend(fontsize=13)
        ax1.set_title(f'Normalized power spectrum of the {marker1} and {marker2} maps and the ML solutions', fontsize=17)
        
    # Create the second subplot

        ax2 = fig.add_subplot(gs[2, :])
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(3)
        ax2.tick_params(axis='both', labelsize=14, width=2)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.scatter(ellw,resw,label=f'{map1.shape[0]}-->{nside1} Percentage residuals white noise',c='orange',)
        ax2.scatter(ellr,resr,label=f'{map1.shape[0]}-->{nside1} Percentage residuals red noise',c='orange',marker='x')
        ax2.scatter(ellw2,resw2,label=f'{map2.shape[0]}-->{nside2} Percentage residuals white noise',c='blue')
        ax2.scatter(ellr2,resr2,label=f'{map2.shape[0]}-->{nside2} Percentage residuals red noise',c='blue',marker='x')
        ax2.axhline(0, color='r', lw=1, ls='--')
        ax2.set_xlabel('$\ell$',fontsize=14)
        ax2.set_ylabel('Percentage residuals',fontsize=14)
        ax2.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        plt.tight_layout()
        plt.savefig(f'{directory}/{marker1}_{marker2}_ResidualPS.png')
        plt.show()
        print(f"The residual power spectrum has been calculated successfully! ----> {marker1}_{marker2}_ResidualPS.png")

    else:
        raise ValueError("The noise parameter should be 0 or 1")
else:
    print("We have 3 input maps!")
    print(f"The angular extent of the input maps is {ang_ext} degrees")
    # Power spectrum of the pysm3 maps
    ellm,dlm=slv.ps_calc(map1,ang_ext,delta_ell=bin_size)
    ellm2,dlm2=slv.ps_calc(map2,ang_ext,delta_ell=bin_size)
    ellm3,dlm3=slv.ps_calc(map3,ang_ext,delta_ell=bin_size)
    print("Calculating the power spectrum of the input maps...")
    print(f"The binning of the power spectrum is $\Delta \ell$ = {binning}")

    # Create a figure
    fig = plt.figure(figsize=(18, 10))
    # Create a gridspec
    gs = gridspec.GridSpec(2, 3, height_ratios=[1,1])

    # Create the first subplot for the power spectrum
    ax1 = fig.add_subplot(gs[1, :])
    # making the spines thicker
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(3)
    ax1.semilogy(ellm, dlm, 'r', label=f'{marker1} map shape {map1.shape[0]}x{map1.shape[1]}')
    ax1.semilogy(ellm2, dlm2, 'b', label=f'{marker2} map shape {map2.shape[0]}x{map2.shape[1]}')
    ax1.semilogy(ellm3, dlm3, 'g', label=f'{marker3} map shape {map3.shape[0]}x{map3.shape[1]}')
    ax1.set_xlabel('$\ell$', fontsize=17)
    ax1.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
    # making the ticks thicker
    ax1.tick_params(axis='both', which='major', labelsize=14, width=2)
    ax1.legend(fontsize=13)

    # Create the second subplot for the map
    ax2 = fig.add_subplot(gs[0, 0])
    im = ax2.imshow(map1, cmap='RdBu', interpolation='bilinear')
    cbar = fig.colorbar(im, ax=ax2, orientation='vertical')
    cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
    ax2.set_title(f'{marker1} map', fontsize=17)

    # Create the third subplot for the map
    ax3 = fig.add_subplot(gs[0, 1])
    im = ax3.imshow(map2, cmap='RdBu', interpolation='bilinear')
    cbar = fig.colorbar(im, ax=ax3, orientation='vertical')
    cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
    ax3.set_title(f'{marker2} map 2', fontsize=17)

    # Create the fourth subplot for the map
    ax4 = fig.add_subplot(gs[0, 2])
    im = ax4.imshow(map3, cmap='RdBu', interpolation='bilinear')
    cbar = fig.colorbar(im, ax=ax4, orientation='vertical')
    cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
    ax4.set_title(f'{marker3} map 3', fontsize=17)
    
    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig(f'{directory}/{marker1}_{marker2}_{marker3}_map_PS.png')
    plt.show()
    print(f"Power spectrum of the input maps has been calculated successfully! ----> {marker1}_{marker2}_{marker3}_map_PS.png")
    # Create the MLsolver object
    nside1=int(map1.shape[0]/4)
    nside2=int(map2.shape[0]/4)
    nside3=int(map3.shape[0]/4)
    mlsolver = slv.MLsolver(map1, nside1)
    mlsolver2 = slv.MLsolver(map2, nside2)
    mlsolver3 = slv.MLsolver(map3, nside3)
    print("Creating the MLsolver objects...")

    if noise==0:
        print("You chose not to add noise to the input maps!")
        sol=mlsolver.ml_solver(0,add_noise=False)
        sol2=mlsolver2.ml_solver(0,add_noise=False)
        sol3=mlsolver3.ml_solver(0,add_noise=False)
        ell,dl=slv.ps_calc(sol,ang_ext,delta_ell=bin_size)
        ell2,dl2=slv.ps_calc(sol2,ang_ext,delta_ell=bin_size)
        ell3,dl3=slv.ps_calc(sol3,ang_ext,delta_ell=bin_size)
        # Create a figure
        fig = plt.figure(figsize=(18, 10))
        # Create a gridspec
        gs = gridspec.GridSpec(2, 3, height_ratios=[1,1])

        # Create the first subplot for the power spectrum
        ax1 = fig.add_subplot(gs[1, :])
        # making the spines thicker
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(3)
        ax1.semilogy(ell, dl, 'r', label=f'ML solution {sol.shape[0]}x{sol.shape[1]}')
        ax1.semilogy(ell2, dl2, 'b', label=f'ML solution {sol2.shape[0]}x{sol2.shape[1]}')
        ax1.semilogy(ell3, dl3, 'g', label=f'ML solution {sol3.shape[0]}x{sol3.shape[1]}')
        ax1.set_xlabel('$\ell$', fontsize=17)
        ax1.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        # making the ticks thicker
        ax1.tick_params(axis='both', which='major', labelsize=14, width=2)
        ax1.legend(fontsize=13)

        # Create the second subplot for the map
        ax2 = fig.add_subplot(gs[0, 0])
        im = ax2.imshow(sol, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax2, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax2.set_title(f'ML solution {marker1} map', fontsize=17)

        # Create the third subplot for the map
        ax3 = fig.add_subplot(gs[0, 1])
        im = ax3.imshow(sol2, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax3, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax3.set_title(f'ML solution {marker2} map', fontsize=17)

        # Create the fourth subplot for the map
        ax4 = fig.add_subplot(gs[0, 2])
        im = ax4.imshow(sol3, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax4, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax4.set_title(f'ML solution {marker3} map', fontsize=17)

        # Adjust the layout and save the figure
        plt.tight_layout()
        plt.savefig(f'{directory}/{marker1}_{marker2}_{marker3}_Solution_without_noise.png')
        plt.show()
        print(f"The ML solutions and their power spectrum have been calculated successfully! ----> {marker1}_{marker2}_{marker3}_Solution_without_noise.png")

        # plotting the comparision power spectrum
        plt.figure(figsize=(18,7))
        plt.semilogy(ellm,dlm,label=f'input map 1 {map1.shape[0]}x{map1.shape[1]}',ls='-.',c='black')
        plt.semilogy(ell,dl,label=f'ML solution 1 {sol.shape[0]}x{sol.shape[1]}',c='black')
        plt.semilogy(ellm2,dlm2,label=f'input map 2 {map2.shape[0]}x{map2.shape[1]}',ls='-.',c='blue')
        plt.semilogy(ell2,dl2,label=f'ML solution 2 {sol2.shape[0]}x{sol2.shape[1]}',c='blue')
        plt.semilogy(ellm3,dlm3,label=f'input map 3 {map3.shape[0]}x{map3.shape[1]}',ls='-.',c='green')
        plt.semilogy(ell3,dl3,label=f'ML solution 3 {sol3.shape[0]}x{sol3.shape[1]}',c='green')
        ax = plt.gca()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        plt.xlabel('$\ell$', fontsize=17)
        plt.ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize=13)
        plt.title('Power spectrum of the input maps and the ML solutions', fontsize=17)
        plt.savefig(f'{directory}/{marker1}_{marker2}_{marker3}_ComparisionPS.png')
        plt.show()
        print(f"The comparision power spectrum has been calculated successfully! ----> {marker1}_{marker2}_{marker3}_ComparisionPS.png")
        print("The residual power spectrum is being calculated...")
        # normalizing the residual PS
        norm=dl/np.mean(dl)
        normm=dlm/np.mean(dlm)
        norm2=dl2/np.mean(dl2)
        normm2=dlm2/np.mean(dlm2)
        norm3=dl3/np.mean(dl3)
        normm3=dlm3/np.mean(dlm3)
        # calculating the residual PS
        res=(norm-normm)/normm * 100
        res2=(norm2-normm2)/normm2 * 100
        res3=(norm3-normm3)/normm3 * 100
        #plotting the residual PS
        plt.figure(figsize=(18,7))
        # Create a figure
        fig = plt.figure(figsize=(18,10))

    # Define the grid
        gs = gridspec.GridSpec(3, 1)

    # Create the first subplot
        ax1 = fig.add_subplot(gs[:2, :])
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(3)
        ax1.tick_params(axis='both', labelsize=14, width=2)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.plot(ell,norm,label=f'{nside1}x{nside1} ML map 1',c='orange')
        ax1.plot(ellm,normm,label=f'{map1.shape[0]}x{map1.shape[1]} {marker1} map',ls='-.',c='black')
        ax1.plot(ell2,norm2+5,label=f'{nside2}x{nside2} ML map 2',c='blue')
        ax1.plot(ellm2,normm2+5,label=f'{map2.shape[0]}x{map2.shape[1]} {marker2} map',ls='-.',c='blue')
        ax1.plot(ell3,norm3+10,label=f'{nside3}x{nside3} ML map 3',c='green')
        ax1.plot(ellm3,normm3+10,label=f'{map3.shape[0]}x{map3.shape[1]} {marker3} map',ls='-.',c='green')
        ax1.set_xlabel('$\ell$', fontsize=17)
        ax1.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        ax1.legend(fontsize=13)
        ax1.set_title('Normalized power spectrum of the input maps and the ML solutions', fontsize=17)

    # Create the second subplot
        ax2 = fig.add_subplot(gs[2, :])
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(3)
        ax2.tick_params(axis='both', labelsize=14, width=2)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.scatter(ell,res,label=f'{map1.shape[0]}-->{nside1} Percentage residuals',c='orange')
        ax2.scatter(ell2,res2,label=f'{map2.shape[0]}-->{nside2} Percentage residuals',c='blue')
        ax2.scatter(ell3,res3,label=f'{map3.shape[0]}-->{nside3} Percentage residuals',c='green')
        ax2.axhline(0, color='r', lw=1, ls='--')
        ax2.set_xlabel('$\ell$',fontsize=14)
        ax2.set_ylabel('Percentage residuals',fontsize=14)
        ax2.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        plt.tight_layout()
        plt.savefig(f'{directory}/{marker1}_{marker2}_{marker3}_ResidualPS.png')
        plt.show()
        print(f"The residual power spectrum has been calculated successfully! ----> {marker1}_{marker2}_{marker3}_ResidualPS.png")

    elif noise==1:
        print("You chose to add noise to the input maps!")
        solw=mlsolver.ml_solver(0,add_noise=True)
        solr=mlsolver.ml_solver(-3.5,add_noise=True)
        solw2=mlsolver2.ml_solver(0,add_noise=True)
        solr2=mlsolver2.ml_solver(-3.5,add_noise=True)
        solw3=mlsolver3.ml_solver(0,add_noise=True)
        solr3=mlsolver3.ml_solver(-3.5,add_noise=True)
        ellw,dlw=slv.ps_calc(solw,ang_ext,delta_ell=bin_size)
        ellr,dlr=slv.ps_calc(solr,ang_ext,delta_ell=bin_size)
        ellw2,dlw2=slv.ps_calc(solw2,ang_ext,delta_ell=bin_size)
        ellr2,dlr2=slv.ps_calc(solr2,ang_ext,delta_ell=bin_size)
        ellw3,dlw3=slv.ps_calc(solw3,ang_ext,delta_ell=bin_size)
        ellr3,dlr3=slv.ps_calc(solr3,ang_ext,delta_ell=bin_size)
       
        # Create a figure
        fig = plt.figure(figsize=(14, 19))

        # Define the grid
        gs = gridspec.GridSpec(4, 2)

        # Create the first subplot for the map
        ax1 = plt.subplot(gs[0, 0])
        im = ax1.imshow(solw, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax1, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax1.set_title('ML solution 1 (W.N.)', fontsize=17)

        # Create the second subplot for the map
        ax2 = plt.subplot(gs[0, 1])
        im = ax2.imshow(solr, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax2, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax2.set_title('ML solution 1 (R.N.)', fontsize=17)

        # Create the third subplot for the map
        ax3 = plt.subplot(gs[1,0])
        im = ax3.imshow(solw2, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax3, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax3.set_title('ML solution 2 (W.N.)', fontsize=17)

        # Create the fourth subplot for the map
        ax4 = plt.subplot(gs[1, 1])
        im = ax4.imshow(solr2, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax4, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax4.set_title('ML solution 2 (R.N.)', fontsize=17)

        # Create the fifth subplot for the map
        ax5 = plt.subplot(gs[2, 0])
        im = ax5.imshow(solw3, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax5, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax5.set_title('ML solution 3 (W.N.)', fontsize=17)

        # Create the sixth subplot for the map
        ax6 = plt.subplot(gs[2, 1])
        im = ax6.imshow(solr3, cmap='RdBu', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax6, orientation='vertical')
        cbar.set_label('temperature [uK]', rotation=270, labelpad=15, fontsize=14)
        ax6.set_title('ML solution 3 (R.N.)', fontsize=17)

        # Create the seventh subplot for the power spectrum
        ax7 = plt.subplot(gs[3, :])
        ax7.semilogy(ellw, dlw, 'r', label=f'ML solution 1 {solw.shape[0]}x{solw.shape[1]} (W.N.)')
        ax7.semilogy(ellr, dlr, 'b', label=f'ML solution 1 {solr.shape[0]}x{solr.shape[1]} (R.N.)')
        ax7.semilogy(ellw2, dlw2, 'g', label=f'ML solution 2 {solw2.shape[0]}x{solw2.shape[1]} (W.N.)')
        ax7.semilogy(ellr2, dlr2, 'y', label=f'ML solution 2 {solr2.shape[0]}x{solr2.shape[1]} (R.N.)')
        ax7.semilogy(ellw3, dlw3, 'c', label=f'ML solution 3 {solw3.shape[0]}x{solw3.shape[1]} (W.N.)')
        ax7.semilogy(ellr3, dlr3, 'm', label=f'ML solution 3 {solr3.shape[0]}x{solr3.shape[1]} (R.N.)')
        ax7.set_xlabel('$\ell$', fontsize=17)
        ax7.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        ax7.tick_params(axis='both', which='major', labelsize=14)
        ax7.legend(fontsize=13)
        ax7.set_title(f'Power spectrum of the ML maps', fontsize=17)

        # Adjust the layout and save the figure
        plt.tight_layout()
        plt.savefig(f'{directory}/{marker1}_{marker2}_{marker3}_Solutions.png')
        plt.show()
        print(f"The ML solutions and their power spectrum have been calculated successfully! ----> {marker1}_{marker2}_{marker3}_Solutions.png")

        # plotting the comparision power spectrum
        plt.figure(figsize=(18,7))
        plt.semilogy(ellm,dlm,label=f'{marker1} map {map1.shape[0]}x{map1.shape[1]}',ls='-.',c='black')
        plt.semilogy(ellw,dlw,label=f'ML solution 1 {solw.shape[0]}x{solw.shape[1]} (W.N.)',c='r')
        plt.semilogy(ellr,dlr,label=f'ML solution 1 {solr.shape[0]}x{solr.shape[1]} (R.N.)',c='b')
        plt.semilogy(ellm2,dlm2,label=f'{marker2} map {map2.shape[0]}x{map2.shape[1]}',ls='-.',c='blue')
        plt.semilogy(ellw2,dlw2,label=f'ML solution 2 {solw2.shape[0]}x{solw2.shape[1]} (W.N.)',c='g')
        plt.semilogy(ellr2,dlr2,label=f'ML solution 2 {solr2.shape[0]}x{solr2.shape[1]} (R.N.)',c='y')
        plt.semilogy(ellm3,dlm3,label=f'{marker3} map {map3.shape[0]}x{map3.shape[1]}',ls='-.',c='green')
        plt.semilogy(ellw3,dlw3,label=f'ML solution 3 {solw3.shape[0]}x{solw3.shape[1]} (W.N.)',c='c')
        plt.semilogy(ellr3,dlr3,label=f'ML solution 3 {solr3.shape[0]}x{solr3.shape[1]} (R.N.)',c='m')
        ax = plt.gca()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        plt.xlabel('$\ell$', fontsize=17)
        plt.ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize=13)
        plt.title('Power spectrum of the input maps and the ML solutions', fontsize=17)
        plt.savefig(f'{directory}/{marker1}_{marker2}_{marker3}_ComparisionPS.png')
        plt.show()
        print(f"The comparision power spectrum has been calculated successfully! ----> {marker1}_{marker2}_{marker3}_ComparisionPS.png")
        print("The residual power spectrum is being calculated...")
        # normalizing the residual PS
        normw=dlw/np.mean(dlw)
        normr=dlr/np.mean(dlr)
        normm=dlm/np.mean(dlm)
        normw2=dlw2/np.mean(dlw2)
        normr2=dlr2/np.mean(dlr2)
        normm2=dlm2/np.mean(dlm2)
        normw3=dlw3/np.mean(dlw3)
        normr3=dlr3/np.mean(dlr3)
        normm3=dlm3/np.mean(dlm3)
        # calculating the residual PS
        resw=(normw-normm)/normm * 100
        resr=(normr-normm)/normm * 100
        resw2=(normw2-normm2)/normm2 * 100
        resr2=(normr2-normm2)/normm2 * 100
        resw3=(normw3-normm3)/normm3 * 100
        resr3=(normr3-normm3)/normm3 * 100
        #plotting the residual PS
        plt.figure(figsize=(18,7))
        # Create a figure
        fig = plt.figure(figsize=(18,10))

    # Define the grid
        gs = gridspec.GridSpec(3, 1)

    # Create the first subplot
        
        ax1 = fig.add_subplot(gs[:2, :])
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(3)
        ax1.tick_params(axis='both', labelsize=14, width=2)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.tick_params(axis='x', labelsize=14)
        ax1.plot(ellw,normw,label=f'{nside1}x{nside1} ML map 1 (w.n. model)',c='orange')
        ax1.plot(ellm,normm,label=f'{map1.shape[0]}x{map1.shape[1]} {marker1} map',ls='-.',c='black')
        ax1.plot(ellr,normr,label=f'{nside1}x{nside1} ML map 1 (r.n. model)',c='orange',ls='--')
        ax1.plot(ellw2,normw2+5,label=f'{nside2}x{nside2} ML map 2 (w.n. model)',c='blue')
        ax1.plot(ellm2,normm2+5,label=f'{map2.shape[0]}x{map2.shape[1]} {marker2} map',ls='-.',c='blue')
        ax1.plot(ellr2,normr2+5,label=f'{nside2}x{nside2} ML map 2 (r.n. model)',c='blue',ls='--')
        ax1.plot(ellw3,normw3+10,label=f'{nside3}x{nside3} ML map 3 (w.n. model)',c='green')
        ax1.plot(ellm3,normm3+10,label=f'{map3.shape[0]}x{map3.shape[1]} {marker3} map',ls='-.',c='green')
        ax1.plot(ellr3,normr3+10,label=f'{nside3}x{nside3} ML map 3 (r.n. model)',c='green',ls='--')
        ax1.set_xlabel('$\ell$', fontsize=17)
        ax1.set_ylabel('$D_{\ell}$ [$\mu$K$^2$]', fontsize=17)
        ax1.legend(fontsize=13)
        ax1.set_title('Normalized power spectrum of the input maps and the ML solutions', fontsize=17)

    # Create the second subplot
        
        ax2 = fig.add_subplot(gs[2, :])
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(3)
        ax2.tick_params(axis='both', labelsize=14, width=2)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='x', labelsize=14)
        ax2.scatter(ellw,resw,label=f'{map1.shape[0]}-->{nside1} Percentage residuals white noise',c='orange',)
        ax2.scatter(ellr,resr,label=f'{map1.shape[0]}-->{nside1} Percentage residuals red noise',c='orange',marker='x')
        ax2.scatter(ellw2,resw2,label=f'{map2.shape[0]}-->{nside2} Percentage residuals white noise',c='blue')
        ax2.scatter(ellr2,resr2,label=f'{map2.shape[0]}-->{nside2} Percentage residuals red noise',c='blue',marker='x')
        ax2.scatter(ellw3,resw3,label=f'{map3.shape[0]}-->{nside3} Percentage residuals white noise',c='green')
        ax2.scatter(ellr3,resr3,label=f'{map3.shape[0]}-->{nside3} Percentage residuals red noise',c='green',marker='x')
        ax2.axhline(0, color='r', lw=1, ls='--')
        ax2.set_xlabel('$\ell$',fontsize=14)
        ax2.set_ylabel('Percentage residuals',fontsize=14)
        ax2.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        plt.tight_layout()
        plt.savefig(f'{directory}/{marker1}_{marker2}_{marker3}_ResidualPS.png')
        plt.show()
        print(f"The residual power spectrum has been calculated successfully! ----> {marker1}_{marker2}_{marker3}_ResidualPS.png")
    else:
        raise ValueError("The noise parameter should be 0 or 1")
print("The code has been executed successfully!")

end_time=time.time()
print(f"Your analysis took {end_time-start_time} seconds!!!")

