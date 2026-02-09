import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from sklearn.metrics import mean_squared_error, mean_absolute_error

import xgboost as xgb
from indago import PSO


import joblib
from sklearn.metrics import *
from sklearn.utils import shuffle

import sys

sys.path.insert(0, 'ALPSOptimizer/src')
from ALPS import ALPS


plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})



class inverseDesignAnalysis:
    
    def __init__(self, train_data_r, 
                       test_data_r, train_data_t, 
                       test_data_t, p, n, wavelengths,
                       inverse_design_model=None):
        
        self.train_data_r = train_data_r #train data tuple reflectance
        self.train_data_t = train_data_t #train data tuple transmittance
        
        self.test_data_r = test_data_r #test_data tuple reflectance
        self.test_data_t = test_data_t #test_data tuple transmittance
        
        self.p = p #percentage of train data used for forward model training
        self.n = n #number of top predictions
        self.wavelengths = wavelengths #wavelength file        
        
        self.inverse_design_model = inverse_design_model #load pretrained models for inverse design
    
    
    def experimental_dataset(self):
                
        laser_parameters_train = self.train_data_r[0]
        laser_parameters_test = self.test_data_r[0]
        
        transmittance_train = self.train_data_t[1]
        transmittance_test = self.test_data_t[1]
        
        reflectance_train = self.train_data_r[1]
        reflectance_test = self.test_data_r[1]
                
        laser_parameters = np.vstack((laser_parameters_train, laser_parameters_test))
        reflectance = np.vstack((reflectance_train, reflectance_test))*100
        transmittance = np.vstack((transmittance_train, transmittance_test))*100
        
        
        absorbance = 100 - transmittance - reflectance
        target_absorbance = np.ones(len(self.wavelengths))*100

        epsilon = np.array([np.sqrt(mean_squared_error(absorbance[i], target_absorbance)) for i in range(len(absorbance))])
        sorted_epsilon = np.argmin(epsilon)

        laser_parameters_sorted = laser_parameters[sorted_epsilon, :]
        absorbance_curves = absorbance[sorted_epsilon, :]

        
        return laser_parameters, absorbance

    def find_test_set_parameters(self, laser_parameters_sorted):
        
        
        laser_parameters_test = self.test_data_r[0]
        transmittance_test = self.test_data_t[1]*100
        reflectance_test = self.test_data_r[1]*100
        
        
        absorbance_test = 100 - transmittance_test - reflectance_test
        
        epsilon_index = []
        

        for i in range(len(laser_parameters_sorted)):
            temp_rmse = []
            index = []

            for j in range(len(laser_parameters_test)):
                rmse_ = np.sqrt(mean_squared_error(laser_parameters_sorted[i], laser_parameters_test[j]))
                temp_rmse.append(rmse_)
                index.append(j)
            
            indx = np.argmin(temp_rmse)
            epsilon_index.append(np.array(index)[indx])

            
        ind_ = np.unique(epsilon_index)
        laser_parameters_ = laser_parameters_test[ind_]
        absorbance_ = absorbance_test[ind_]
        
        
        return laser_parameters_, absorbance_
        
        
        
    def train_model(self, data, save_model=False, output='Transmittance'):
        
        X, y = data[0], data[1]
        
        input_train = X[:int(self.p*len(X))]
        output_train = y[:int(self.p*len(y))]
        
        print (len(input_train))
        
        X_train, X_val, y_train, y_val = train_test_split(input_train, 
                                                            output_train,
                                                            test_size=0.1, shuffle=False) 
        
        model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=500, 
                    reg_lambda=2,
                    learning_rate=0.05, 
                    max_depth=5, early_stopping_rounds=5
                    )
        
        model.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_val, y_val)],
                 verbose=False
                 )
        
        
        evals_result = model.evals_result()
        epochs = len(evals_result['validation_0']['rmse'])
        x_axis = range(0, epochs)
        

        plt.figure(figsize=(6,5))
        percetange_ = self.p*100
        plt.title(f'{output}, {percetange_}\% train data')
        
        train_rmse = np.array(evals_result['validation_0']['rmse'])*100
        validation_rmse = np.array(evals_result['validation_1']['rmse'])*100
            
        plt.plot(x_axis, train_rmse, label='Train')
        plt.plot(x_axis, validation_rmse, label='Validation')
        
        plt.legend()
        plt.ylabel('RMSE (\%)')
        plt.xlabel('Estimators')
        plt.ylim(0, 3)
        ax = plt.gca()

        for axis in ['top', 'bottom', 'left', 'right']:
          ax.spines[axis].set_linewidth(2)
        
        plt.tight_layout()
        
        plt.savefig(f'inverse_design_results/loss_curve_{output}_{self.p}_model_result.pdf', dpi=400)
        plt.savefig(f'inverse_design_results/loss_curve_{output}_{self.p}_model_result.eps', dpi=400)
        plt.savefig(f'inverse_design_results/loss_curve_{output}_{self.p}_model_result.png', dpi=400)

        if save_model == True:
            joblib.dump(f'inverse_design_models/{output}_model_{self.p}.pkl')
        
        return model
    
    
    def get_models(self, save_model=False):
        
        if self.inverse_design_model != None:
            model_t, model_r = self.inverse_design_model[0], self.inverse_design_model[1]
        else:
            print ("Transmittance model training")
            model_t = self.train_model(self.train_data_t, output='Transmittance')
            print ("Reflectance model training")
            model_r = self.train_model(self.train_data_r, output='Reflectance')
            
        return model_t, model_r
    
    
    def get_plot_predictions(self):
        
        transmittance_model, reflectance_model = self.get_models()
        
        """Generating an absorbance map"""      
        
        laser_parameters = self.train_data_t[0]
        upper_bounds = np.max(laser_parameters, axis=0)
        lower_bounds = np.min(laser_parameters, axis=0)
        laser_parameters_pdp = np.random.uniform(lower_bounds, upper_bounds, size=(20000, len(lower_bounds)))
        
    
        transmittance_values = transmittance_model.predict(laser_parameters_pdp)
        reflectance_values = reflectance_model.predict(laser_parameters_pdp)
        
        
        absorbance_values = 1 - transmittance_values - reflectance_values 
        mean_absorbance = np.mean(absorbance_values, axis=1)*100
        

        
        
        
        
        # """PDP plot"""
        
        # plt.figure(figsize=(6,5))
        # contour = plt.scatter(laser_parameters_pdp[:, 0], laser_parameters_pdp[:,1], c=mean_absorbance, cmap='plasma')
        # cbar = plt.colorbar(contour)
        # cbar.set_label('Mean absorbance (\%)')#, rotation=270, labelpad=15)

        # rect = Rectangle((299, 0), 201, 43, linewidth=1, edgecolor='none', facecolor='white')

        # plt.gca().add_patch(rect)
        # plt.xlabel(r'Speed ($\upmu$m/s)')
        # plt.ylabel(r'Spacing ($\upmu$m)')

        # ax = plt.gca()

        # for axis in ['top', 'bottom', 'left', 'right']:
        #   ax.spines[axis].set_linewidth(2)

        # plt.tight_layout()
        
        

        # """Experimental data plot"""
        
        # laser_parameters_exp, absorbance_exp = self.experimental_dataset()
                
        # plt.figure(figsize=(6,5))
        # contour = plt.scatter(laser_parameters_exp[:, 0], laser_parameters_exp[:,1], c=np.mean(absorbance_exp, axis=1), cmap='plasma')
        # cbar = plt.colorbar(contour)
        # cbar.set_label('Mean absorbance (\%)')#, rotation=270, labelpad=15)

        # # rect = Rectangle((299, 0), 201, #res:ner-page=543, linewidth=1, edgecolor='none', facecolor='white')

        # # plt.gca().add_patch(rect)
        # plt.xlabel(r'Speed ($\upmu$m/s)')
        # plt.ylabel(r'Spacing ($\upmu$m)')

        # ax = plt.gca()

        # for axis in ['top', 'bottom', 'left', 'right']:
        #   ax.spines[axis].set_linewidth(2)

        # plt.tight_layout()

    

        """Inverse design - measuring discrepancy based on the 100% absorbance target"""
        
        target_absorbance = np.ones(len(self.wavelengths))

        epsilon = np.array([np.sqrt(mean_squared_error(absorbance_values[i], target_absorbance)) for i in range(len(absorbance_values))])
        sorted_epsilon = np.argsort(epsilon)[:self.n]
        epsilon_values = epsilon[sorted_epsilon]
        print ('rs', epsilon_values)

        laser_parameters_sorted = laser_parameters_pdp[sorted_epsilon, :]
        absorbance_curves = absorbance_values[sorted_epsilon, :]
        
        laser_parameters_sorted = []
        eps_val = []
        for _ in range(self.n):
        
            def f(x):
                
                transmittance_values = transmittance_model.predict([x])
                reflectance_values = reflectance_model.predict([x])
                
                absorbance_eval = 1 - transmittance_values[0] - reflectance_values[0]
                
                epsilon = np.sqrt(mean_squared_error(target_absorbance, absorbance_eval))
                
                
                return epsilon, absorbance_eval.reshape(1,-1)

            test_full = ALPS(target=target_absorbance, 
            max_evals=400, 
            initial_sample_size=5, 
            model_batch_size=5, 
            lb=lower_bounds, 
            ub=upper_bounds, 
            experiment=f, sample_size=100)

            _, resultz, fitness_  = test_full.run(1)
            laser_parameters_sorted.append(resultz)
            eps_val.append(fitness_)
            
        laser_parameters_sorted = np.array(laser_parameters_sorted)
        print ('epsilon value', np.sort(eps_val))
        percentage_ = self.p*100

        # plt.plot(laser_parameters_sorted[:, 0], laser_parameters_sorted[:,1], 'kx')
        
        plt.figure(figsize=(6,5))
        for i in range(len(absorbance_curves)):
            speed = laser_parameters_sorted[i, 0]
            spacing = laser_parameters_sorted[i, 1]
            plt.plot(wavelengths, absorbance_curves[i,:]*100, alpha=0.5, linewidth=3, label=f'{speed:.2f}' + r' $\upmu$m/s,' + f' {spacing:.2f}' + r' $\upmu$m')
                       
        plt.xlabel(r'Wavelength ($\upmu$m)')
        plt.ylabel('Absorbance (\%)')
        plt.title('%0.f' % percentage_ + '\% train data, model prediction')
        plt.ylim(0, 100)
        plt.legend()

        ax = plt.gca()

        for axis in ['top', 'bottom', 'left', 'right']:
          ax.spines[axis].set_linewidth(2)

        plt.tight_layout()
        plt.savefig(f'inverse_design_results/absorbance_curves_{percentage_}_prediction_result.pdf', dpi=400)
        plt.savefig(f'inverse_design_results/absorbance_curves_{percentage_}_prediction_result.eps', dpi=400)
        plt.savefig(f'inverse_design_results/absorbance_curves_{percentage_}_prediction_result.png', dpi=400)
        
        
        """Experimental validation absorbance plot"""
        
        laser_parameters_, absorbance_ = self.find_test_set_parameters(laser_parameters_sorted)
        
        plt.figure(figsize=(6,5))
        for i in range(len(absorbance_)):

            # plt.plot(self.wavelengths, absorbance_[i,:], 'k-')
            # lp = laser_parameters_[i]
            speed = laser_parameters_[i, 0]
            spacing = laser_parameters_[i, 1]
            plt.plot(self.wavelengths, absorbance_[i,:], alpha=0.5, linewidth=3, label=r'%0.f $\upmu$m/s, %0.f $\upmu$m' % (speed, spacing))

            
            
        # plt.plot(self.wavelengths, absorbance_[-1,:], label='Experiment')
        # lp_last = laser_parameters_[-1]
        # plt.plot(self.wavelengths, absorbance_[-1,:], label=r'{speed} $\upmu$m/s, {spacing} $\upmu$m')

            
        # for i in range(len(absorbance_curves)-1):

        #     plt.plot(self.wavelengths, absorbance_curves[i,:]*100, 'r-')
            
        # plt.plot(self.wavelengths, absorbance_curves[-1,:]*100, 'r-', label='Prediction')
        
        
        plt.xlabel(r'Wavelength ($\upmu$m)')
        plt.ylabel('Absorbance (\%)')
        plt.legend()
        # plt.title('%0.f \% train data, experimental validation' % percentage_)
        plt.title('%0.f' % percentage_ + '\% train data, experimental validation')

        plt.ylim(0, 100)

        ax = plt.gca()

        for axis in ['top', 'bottom', 'left', 'right']:
          ax.spines[axis].set_linewidth(2)

        plt.tight_layout()
        
        plt.savefig(f'inverse_design_results/absorbance_curves_{percentage_}_model_result.pdf', dpi=400)
        plt.savefig(f'inverse_design_results/absorbance_curves_{percentage_}_model_result.eps', dpi=400)
        plt.savefig(f'inverse_design_results/absorbance_curves_{percentage_}_model_result.png', dpi=400)
        
        
        """Experimental data plot"""
        
        laser_parameters_exp, absorbance_exp = self.experimental_dataset()
                
        plt.figure(figsize=(6,5))
        contour = plt.scatter(laser_parameters_exp[:, 0], laser_parameters_exp[:,1], c=np.mean(absorbance_exp, axis=1), cmap='plasma')
        plt.plot(laser_parameters_[:, 0], laser_parameters_[:,1], 'kx')#, label='Experiment')
        

        plt.plot(laser_parameters_sorted[:, 0], laser_parameters_sorted[:,1], 'rx')#, label='Prediction')
        plt.legend(loc='best', framealpha=0, fontsize=12)

        cbar = plt.colorbar(contour)
        cbar.set_label('Mean absorbance (\%)')#, rotation=270, labelpad=15)

        plt.xlabel(r'Speed ($\upmu$m/s)')
        plt.ylabel(r'Spacing ($\upmu$m)')
        plt.title(f'{percentage_}\% train data')

        ax = plt.gca()

        for axis in ['top', 'bottom', 'left', 'right']:
          ax.spines[axis].set_linewidth(2)
          
        plt.tight_layout()
                
        plt.savefig(f'inverse_design_results/laser_parameters_{percentage_}_model_result.pdf', dpi=400)
        plt.savefig(f'inverse_design_results/laser_parameters_{percentage_}_model_result.eps', dpi=400)
        plt.savefig(f'inverse_design_results/laser_parameters_{percentage_}_model_result.png', dpi=400)
        
        
k = 10 
laser_parameters_train = np.load('train_data/laser_parameters_train.npy')
reflectance_train = np.load('train_data/reflectance_train.npy')[:, ::k]/100
transmittance_train = np.load('train_data/transmittance_train.npy')[:, ::k]/100   

laser_parameters_test = np.load('test_data/laser_parameters_test.npy')
reflectance_test = np.load('test_data/reflectance_test.npy')[:, ::k]/100   
transmittance_test = np.load('test_data/transmittance_test.npy')[:, ::k]/100   

wavelengths = np.load('data/wavelengths.npy')[::k]


train_data_r = (laser_parameters_train, reflectance_train)
train_data_t = (laser_parameters_train, transmittance_train)

test_data_r = (laser_parameters_test, reflectance_test)
test_data_t = (laser_parameters_test, transmittance_test)


p=1 #fraction of train dataset used, i.e. 1 = 100%
n=5 #top n predictions

test = inverseDesignAnalysis(train_data_r, test_data_r,
                             train_data_t, test_data_t, p, n,
                             wavelengths=wavelengths)

test.get_plot_predictions()

    
            



