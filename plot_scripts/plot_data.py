import matplotlib.pyplot as plt
import numpy as np
import os

# Create the directory if it does not exist
output_dir = 'visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Update matplotlib parameters without using LaTeX
plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})

class PlotData:
    
    def __init__(self, laser_params, reflectance, transmittance, wavelength):
        self.laser_params = laser_params
        self.speeds = laser_params[:, 0]
        self.spacings = laser_params[:, 1]
        self.reflectance = reflectance
        self.transmittance = transmittance
        self.absorbance = 100 - self.reflectance - self.transmittance
        self.wavelength = wavelength
    
    def save_plot(self, filename):
        ax = plt.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        
        files = ['.pdf', '.eps']
        for fe in files:
            plt.savefig(os.path.join(output_dir, filename+f'{fe}'), dpi=300)
        plt.close()

    def plot_average_vs_param(self, values, param_name, ylabel, filename):
        
        if param_name == "Speed":
            param = self.speeds
            units = r" (mm/s)"
        elif param_name == "Spacing":
            param = self.spacings
            units = r" ($\upmu$m)"
        averages = []
        for i in values:
            averages.append(np.mean(i))
        plt.figure(figsize=(6,5))
        plt.scatter(param, averages, color='magenta')# color='#7B1B38')
        plt.ylim(73, 93)
        plt.xlabel(param_name + units)
        plt.ylabel('Mean absorptance (\%)')
        plt.tight_layout()
        self.save_plot(filename)

    def plot_wavelength(self, param, param_name, filename, a=0.05, c='red'):
        plt.figure(figsize=(6,5))
        for i in param:
            plt.plot(self.wavelength, i, alpha=a, color=c)
        plt.xlabel(r'Wavelength ($\upmu$m)')
        plt.ylabel(f"{param_name} (\%)")
        if param_name != 'Absorptance':
            plt.ylim(0, 40)
        else:
            plt.ylim(50, 100)
        plt.tight_layout()
        self.save_plot(filename)



    def plot_one_to_many(self, param, filename, laser_parameters):
        # 
        # np.where(self.laser_params == laser_params)
        matches = np.all(self.laser_params[:, None] == laser_params, axis=2)
        idx = np.where(matches)[0]
        
        abs_subset = param[idx, :]
        plt.figure(figsize=(6,5))
        
        # print (laser_params)
        # 
        for i in range(len(abs_subset)):
            # print (i)
            plt.plot(self.wavelength, abs_subset[i], label=r'Speed = %d mm/s, Spacing = %d $\upmu$m' % (laser_params[i, 0], laser_params[i, 1]))
            
        plt.xlabel(r'Wavelength ($\upmu$m)')
        plt.ylabel(f"Absorptance (\%)")
        plt.legend(fontsize=12)

        plt.ylim(50, 100)
        plt.tight_layout()
        self.save_plot(filename)



    def plot_contour(self, values, title, cmap, filename):
        # spacing, speed = np.meshgrid(np.unique(self.spacings), np.unique(self.speeds))
        # z = np.array([np.mean(values[(self.speeds == s) & (self.spacings == sp)]) for s, sp in zip(speed.flatten(), spacing.flatten())])
        # z = z.reshape(speed.shape)
        

        
        plt.figure(figsize=(6,5))
        cp = plt.tricontour(self.speeds, self.spacings, values, levels=np.linspace(50, 100, 8000), cmap=cmap)
        cbar = plt.colorbar(cp)
        # cbar.set_label(r'Absorbance (\%) at wavelength = 2.5 $\upmu$m')
        cbar.set_label(r'Mean absorptance (\%)')

        # cbar.set_ticks(np.arange(60, 100, 2))
        plt.xlabel(r'Speed ($\upmu$m/s)')
        plt.ylabel(r'Spacing ($\upmu$m)')
        # #plt.title(title)
        # plt.show()
        self.save_plot(filename)

    def plot_reflectance_contour(self):
        self.plot_contour(self.reflectance, 'Reflectance Contour Plot', 'plasma', 'reflectance_contour')

    def plot_transmittance_contour(self):
        self.plot_contour(self.transmittance, 'Transmittance Contour Plot', 'plasma', 'transmittance_contour')

    def plot_absorbance_contour(self):
        self.plot_contour(self.absorbance, 'Absorbance Contour Plot', 'plasma', 'absorbance_contour')

    def plot_mean_absorbance_contour(self):
        mean_absorbance = np.mean(self.absorbance, axis=1)
        self.plot_contour(mean_absorbance, 'Mean Absorbance Contour Plot', 'plasma', 'mean_absorbance_contour')

    def plot_absorbance_at_wavelength(self, target_wavelength, title, filename):
        idx = (np.abs(self.wavelength - target_wavelength)).argmin()
        absorbance_at_wavelength = self.absorbance[:, idx]
        self.plot_contour(absorbance_at_wavelength, title, 'plasma', filename)

    def plot_absorbance_at_2_5_mu_contour(self):
        self.plot_absorbance_at_wavelength(2.5, 'Absorbance at 2.5 µm Contour Plot', 'absorbance_2_5_mu_contour')

    def plot_absorbance_at_0_83_mu_contour(self):
        self.plot_absorbance_at_wavelength(0.83, 'Absorbance at 0.83 µm Contour Plot', 'absorbance_0_83_mu_contour')


# Assuming that you have already loaded the data from .npy files
k = 20 
laser_params = np.load('../data/laser_parameters_cut.npy')
reflectance = np.load('../data/reflectance_calibrated.npy')[:, ::k]
transmittance = np.load('../data/transmittance_calibrated.npy')[:, ::k]
wavelength = np.load("../data/wavelength_calibrated.npy")[::k]
plot_data = PlotData(laser_params, reflectance, transmittance, wavelength)


laser_params = np.array([[160, 41],[270, 15],[220, 24],[140,42]])

plot_data.plot_one_to_many(100 - reflectance - transmittance, "one_to_many_absorptance", laser_params)

# # Generate the plots
# plot_data.plot_mean_absorbance_contour()
# plot_data.plot_absorbance_at_2_5_mu_contour()
# plot_data.plot_absorbance_at_0_83_mu_contour()

# Wavelength Graphs
# plot_data.plot_wavelength(reflectance, "Reflectance", "reflectance_wavelength")
# plot_data.plot_wavelength(transmittance, "Transmittance", "transmittence_wavelength")
# plot_data.plot_wavelength(100 - reflectance - transmittance, "Absorptance", "absorptance_wavelength")

# # Relational Graphs
# plot_data.plot_average_vs_param(transmittance, "Speed", "Transmittence (\%)", "tramittance_speed")
# plot_data.plot_average_vs_param(reflectance, "Speed", "Reflectance (\%)", "reflectance_speed")
# plot_data.plot_average_vs_param(100 - reflectance - transmittance, "Speed", "Absorptance (\%)", "absorptance_speed")

# plot_data.plot_average_vs_param(transmittance, "Spacings", "Transmittence (\%)", "tramittance_spacings")
# plot_data.plot_average_vs_param(reflectance, "Spacings", "Reflectance (\%)", "reflectance_spacings")
#plot_data.plot_average_vs_param(100 - reflectance - transmittance, "Spacing", "Absorptance (\%)", "absorptance_spacings")

# Contour Plots
