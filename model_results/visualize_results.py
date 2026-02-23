import numpy as np
import matplotlib.pyplot as plt
import joblib
from matplotlib.patches import Patch, Rectangle
# import matplotlib.patches as patches

from sklearn.metrics import r2_score, mean_squared_error

plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})


"""plot r2 absorbance score"""
"""plot predicted vs true curves"""
"""plot model based absorbance"""

frac=1
model='lr'

wavelengths = np.load('../data/wavelength_calibrated.npy')[::40]

predicted_transmittance = np.loadtxt(f'predicted_set_transmittance_{frac}_{model}.dat')
predicted_reflectance = np.loadtxt(f'predicted_set_reflectance_{frac}_{model}.dat')

test_transmittance = np.loadtxt(f'test_set_transmittance_{frac}_{model}.dat')
test_reflectance = np.loadtxt(f'test_set_reflectance_{frac}_{model}.dat')

predicted_absorbance = 100 - predicted_transmittance - predicted_reflectance
predicted_absorbance_mean = np.mean(predicted_absorbance, axis=1)

test_absorbance = 100 - test_transmittance - test_reflectance
test_absorbance_mean = np.mean(test_absorbance, axis=1)


plt.figure(figsize=(6,5))

plt.plot(test_absorbance_mean, predicted_absorbance_mean, 'bo')


r2_ = r2_score(test_absorbance_mean, predicted_absorbance_mean)


reg = np.polyfit(test_absorbance_mean, predicted_absorbance_mean, 1)
reg_model = np.poly1d(reg)
plt.plot(test_absorbance_mean, reg_model(test_absorbance_mean), 'r-')



plt.xticks([82, 84, 86, 88, 90])
# plt.xticks([86, 90, 94])

plt.yticks([82, 84, 86, 88, 90])
# plt.yticks([86, 90, 94])


plt.text(88, 84, f'$R^2$ = {r2_:.2f}', fontsize=18, bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel('Experimental mean absorptance (\%)')
plt.ylabel('Predicted mean absorptance (\%)')
# plt.ylim(76, 90)
# plt.xlim(76, 90)
ax = plt.gca()

for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)

plt.tight_layout()

plt.savefig(f'r2_{model}_{frac}.pdf', dpi=400)   
plt.savefig(f'r2_{model}_{frac}.png', dpi=400)   
plt.savefig(f'r2_{model}_{frac}.eps', dpi=400)   

# import sys
# sys.exit()




# plt.savefig(f'Figure3/r2_score_absorbance.pdf', dpi=400)   
# plt.savefig(f'Figure3/r2_score_absorbance.png', dpi=400)   
# plt.savefig(f'Figure3/r2_score_absorbance.eps', dpi=400)   


plt.figure(figsize=(6,5))

test_absorbance = test_absorbance
predicted_absorbance = predicted_absorbance


print (np.where(predicted_absorbance[:,-1] < 75))



# indx = [0, 44, 11, 23, 12]
# indx = [24, 12, 103]

indx = [14, 11, 18, 2, 54]
k=8#, 5, 6, 7]
position_y = [86.5-6, 81.4-7, 78.2-8, 71-10, 74.5-9]#, 77.5, 72.5, 67.5]
position_x = [2.44, 2.44, 2.44, 2.44, 2.44]#, 2.44, 2.44, 2.44]
for i in range(len(indx)):

    plt.plot(wavelengths, test_absorbance[indx[i],:], 'r-', label=f'{i}')
    plt.plot(wavelengths, predicted_absorbance[indx[i],:], 'b-')#, label='Prediction')
    
    rmse_value = np.sqrt(mean_squared_error(test_absorbance[indx[i],:], predicted_absorbance[indx[i],:]))

    mid_wavelength = (wavelengths[0] + wavelengths[-1]) / 2
    mid_test_absorbance = (test_absorbance[i, 0] + test_absorbance[i, -1]) / 2
    plt.text(position_x[i], position_y[i], f'{rmse_value:.2f} \%', fontsize=12, color='black', ha='center')


plt.xlabel(r'Wavelength ($\upmu$m)')
plt.ylabel('Absorptance (\%)')
plt.legend()
plt.ylim(50, 100)

ax = plt.gca()

for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)

plt.tight_layout()


custom_patch = [Patch(color='blue', label='Prediction'), Patch(color='red', label='Experiment')]
plt.legend(handles=custom_patch, loc='upper right', fontsize=12)

# plt.legend()

# rmse_values = np.array([np.sqrt(mean_squared_error(test_absorbance[i,:], predicted_absorbance[i,:])) for i in indx])


plt.savefig(f'rmse_pairs.pdf', dpi=400)   
plt.savefig(f'rmse_pairs.png', dpi=400)   
plt.savefig(f'rmse_pairs.eps', dpi=400)   




