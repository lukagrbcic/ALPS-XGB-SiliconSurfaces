import numpy as np
import matplotlib.pyplot as plt
import sys

plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})

input_train = np.load('../train_data/laser_parameters_train.npy')

input_test = np.load('../test_data/laser_parameters_test.npy')


reflectance_train = np.load('../train_data/reflectance_train.npy') 
transmittance_train = np.load('../train_data/transmittance_train.npy') 


reflectance_test = np.load('../test_data/reflectance_test.npy') 
transmittance_test = np.load('../test_data/transmittance_test.npy') 


absorbance_test = 100 - transmittance_test - reflectance_test
absorbance_train = 100 - transmittance_train - reflectance_train


input_full = np.vstack((input_test, input_train))
absorbance_full = np.vstack((absorbance_test, absorbance_train))

# laser_parameters_exp = input_train
# absorbance_exp = absorbance_train  


laser_parameters_exp = input_train#input_full#input_train[:int(1*len(input_train))]
absorbance_exp = absorbance_train#absorbance_full#absorbance_train[:int(1*len(input_train))]   
 
plt.figure(figsize=(6,5))
contour = plt.scatter(laser_parameters_exp[:, 0], laser_parameters_exp[:,1], c=np.mean(absorbance_exp, axis=1), cmap='plasma')

# print (laser_parameters_)
# print (laser_parameters_sorted)

plt.legend(loc='best', framealpha=0, fontsize=12)

cbar = plt.colorbar(contour)
cbar.set_label('Mean absorptance (\%)')#, rotation=270, labelpad=15)

# rect = Rectangle((299, 0), 201, 43, linewidth=1, edgecolor='none', facecolor='white')
# plt.gca().add_patch(rect)
plt.xlabel(r'Speed (mm/s)')
plt.ylabel(r'Spacing ($\upmu$m)')
plt.title(f'$N_M$ = 226')

ax = plt.gca()

for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)
  
plt.tight_layout()
        
plt.savefig(f'scatter_plot_226.pdf', dpi=400)
plt.savefig(f'scatter_plot_226.eps', dpi=400)
plt.savefig(f'scatter_plot_226.png', dpi=400)