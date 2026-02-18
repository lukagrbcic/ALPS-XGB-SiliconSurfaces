import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor


import joblib
from sklearn.metrics import *
from sklearn.utils import shuffle


plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})


class model_analysis:
    
    def __init__(self, analysis, data, input_train, output_train, fraction=1, model='xgb'):
        
        self.analysis = analysis
        self.data = data 
        self.input_train = input_train
        self.output_train = output_train
        self.fraction = fraction
        self.model = model

    def get_results(self):
               
        if self.analysis == 'lc':
            print ('learning curve analysis')
            

            k=3
            
            """tested linear regression"""
            if self.model == 'lr':
                model = Pipeline([
                    ('scaler', StandardScaler()),                          # Step to standardize the features
                    ('multi_output_regressor', MultiOutputRegressor(LinearRegression()))  # Step to perform multi-output linear regression
                ])
            
            else:

                
                model = xgb.XGBRegressor(
                            objective='reg:squarederror',
                            n_estimators=500, 
                            reg_lambda=20,
                            learning_rate=0.05, 
                            max_depth=5, early_stopping_rounds=5,
                            #random_state=111
                            )
            
            
            rrmse_mean = []
            rrmse_std = []
            
            data_size = [0.25, 0.5, 0.75, 1]
            
            for size in data_size:
                print ('Data size:', size)

                input_train = self.input_train[:int(size*len(self.input_train))]
                output_train = self.output_train[:int(size*len(self.output_train))]
                
    
    
                temp_rmse = []            
    
                for i in range(k):
                                        
                    
                    X_train, X_test, y_train, y_test = train_test_split(input_train, 
                                                                        output_train,
                                                                        test_size=float(1/k)) 
                    
                    X_train_, X_val, y_train_, y_val = train_test_split(X_train, 
                                                                        y_train,
                                                                        test_size=0.1) 
                    
                    if self.model == 'lr':
                        """this is for linear regression"""
                        model.fit(X_train_, y_train_)
                                  
                    
                    else:
                        model.fit(X_train_, y_train_,
                                eval_set=[(X_train_, y_train_), (X_val, y_val)],
                                verbose=False
                             )
                    
                    
                    y_pred = model.predict(X_test)

                    temp_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                

                    
                rrmse_mean.append(np.mean(temp_rmse))
                rrmse_std.append(np.std(temp_rmse))  
            
            data_size_raw = np.array(data_size)*len(self.input_train)
            data_size_raw = np.array([int(i) for i in data_size_raw])
            results = np.hstack((data_size_raw.reshape(-1,1), 
                                 np.array(data_size).reshape(-1,1),
                                 np.array(rrmse_mean).reshape(-1,1),
                                 np.array(rrmse_std).reshape(-1,1)))
            
            results_dict = {'analysis': 'lc',
                            'data_size_raw': data_size_raw,
                            'data_size': np.array(data_size),
                            'rmse_mean': np.array(rrmse_mean),
                            'rmse_std': np.array(rrmse_std),
                            'data': self.data}
            

            
            rrmse_mean = np.array(rrmse_mean)#*100
            rrmse_std = np.array(rrmse_std)#*100 
            np.savetxt(f'model_results/results_learning_curve_{self.data}_{self.model}.txt', results, fmt='%.3f')
         
        else:
            
            print ('histogram analysis')
            
            if self.model == 'lr':
                model = Pipeline([
                    ('scaler', StandardScaler()),                          # Step to standardize the features
                    ('multi_output_regressor', MultiOutputRegressor(LinearRegression(random_state=111)))  # Step to perform multi-output linear regression
                ])
            
            else:
                

                model = xgb.XGBRegressor(
                            objective='reg:squarederror',
                            n_estimators=500, 
                            reg_lambda=20,
                            learning_rate=0.05, 
                            max_depth=5, early_stopping_rounds=5,
                            random_state=111
                            )
            
             
            X_train, X_test, y_train, y_test = train_test_split(self.input_train, 
                                                                self.output_train,
                                                                test_size=0.3, shuffle=False, random_state=23) 

            X_train_, X_val, y_train_, y_val = train_test_split(X_train[:int(self.fraction*len(X_train))], 
                                                                y_train[:int(self.fraction*len(y_train))],
                                                                test_size=0.1, shuffle=False, random_state=23) 
                       
           
            
           
            
            
            # """this is for linear regression"""
            if self.model == 'lr':
                model.fit(X_train_, y_train_)
            
            else:
                model.fit(X_train_, y_train_,
                        eval_set=[(X_train_, y_train_), (X_val, y_val)],
                        verbose=False
                         )
                
                # Save training curves for XGBoost in histogram analysis
                self.plot_training_curves(model, self.fraction)
                  
 
            
            y_pred = model.predict(X_test)
            
            rmse_list = np.array([np.sqrt(mean_squared_error(y_test[i], y_pred[i])) for i in range(len(y_pred))])
            rmse_mean_list = np.array([np.mean(rmse_list), np.max(rmse_list), np.std(rmse_list)])
            np.savetxt(f'model_results/results_histogram_{self.data}_{self.model}', rmse_mean_list, fmt='%.3f')
            np.savetxt(f'model_results/test_set_{self.data}_{self.fraction}_{self.model}.dat', y_pred, fmt='%.3f')
            np.savetxt(f'model_results/predicted_set_{self.data}_{self.fraction}_{self.model}.dat', y_test, fmt='%.3f')


            results_dict = {'analysis': 'hist', 'rmse_values': rmse_list, 'data': self.data}
            
            
            
        
        return results_dict

    def plot_training_curves(self, model, fraction):
        """Plot and save training and validation loss curves for XGBoost"""
        
        results = model.evals_result()
        train_rmse = np.sqrt(results['validation_0']['rmse'])
        val_rmse = np.sqrt(results['validation_1']['rmse'])
        
        estimators = np.arange(len(train_rmse))
        
        # Determine dataset name
        if self.data == 'transmittance':
            name = 'Transmittance'
        else:
            name = 'Reflectance'
        
        plt.figure(figsize=(6, 5))
        plt.plot(estimators, train_rmse, label='Train', linewidth=2)
        plt.plot(estimators, val_rmse, label='Validation', linewidth=2)
        plt.xlabel('Estimators')
        plt.ylabel('RMSE (\%)')
        plt.title(f'{name}, {fraction*100:.1f}\% train data')
        plt.legend(fontsize=18)
        plt.ylim(0, 3)
        
        ax = plt.gca()
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'model_results/training_curves_{self.data}_{fraction}_{self.model}.pdf', dpi=400)
        plt.savefig(f'model_results/training_curves_{self.data}_{fraction}_{self.model}.png', dpi=400)
        plt.savefig(f'model_results/training_curves_{self.data}_{fraction}_{self.model}.eps', dpi=400)
        plt.close()

        
    def plot_results(self, results):
        
        color = ['red', 'green']
        plt.figure(figsize=(6,5))
        
        if results[0]['analysis'] == 'lc':
            for i in range(len(results)):
                
                rrmse_mean = results[i]['rmse_mean']
                rrmse_std = results[i]['rmse_std']
                data_size = results[i]['data_size']
                data = results[i]['data']
                if data == 'transmittance':
                    name = 'Transmittance'
                else:
                    name = 'Reflectance'
                    
                alpha=0.1
                plt.plot(np.array(data_size)*100, rrmse_mean, color=color[i], marker='o', linestyle='-', label=name)
                plt.fill_between(np.array(data_size)*100, rrmse_mean, rrmse_mean + rrmse_std, color=color[i], alpha=alpha)
                plt.fill_between(np.array(data_size)*100, rrmse_mean, rrmse_mean - rrmse_std, color=color[i], alpha=alpha)
        
                plt.xlabel('Dataset size (\%)')
                plt.ylabel('RMSE (\%)')
                plt.xticks([25, 50, 75, 100])
                plt.ylim(0, 5)
                plt.legend(fontsize=18)
                
                ax = plt.gca()
        
                for axis in ['top', 'bottom', 'left', 'right']:
                  ax.spines[axis].set_linewidth(2)
                
                plt.tight_layout()
            plt.savefig(f'model_results/learning_curve_{data}_{self.model}.pdf', dpi=400)
            plt.savefig(f'model_results/learning_curve_{data}_{self.model}.png', dpi=400)
            plt.savefig(f'model_results/learning_curve_{data}_{self.model}.eps', dpi=400)


        else:
            for i in range(len(results)):
                
                rmse_list = results[i]['rmse_values']#*100
                data = results[i]['data']
                if data == 'transmittance':
                    name = 'Transmittance'
                    color = 'red'
                else:
                    name = 'Reflectance'
                    color = 'green'
                    
                alpha=0.75
                plt.hist(rmse_list, bins=50, color=color, alpha=alpha, label=name)
                
                plt.ylabel('Predictions')
                plt.xlabel('RMSE (\%)')
                # plt.ylim(0, 10)
                plt.xlim(0, 3)
                plt.legend(fontsize=18)
                
                ax = plt.gca()
           
                for axis in ['top', 'bottom', 'left', 'right']:
                  ax.spines[axis].set_linewidth(2)
                
                plt.tight_layout()
        
                plt.savefig(f'model_results/histogram_analysis_{data}_{self.fraction}_{self.model}.pdf', dpi=400)
                plt.savefig(f'model_results/histogram_analysis_{data}_{self.fraction}_{self.model}.png', dpi=400)
                plt.savefig(f'model_results/histogram_analysis_{data}_{self.fraction}_{self.model}.eps', dpi=400)


            
data = 'transmittance'
input_train = np.load('train_data/laser_parameters_train.npy')
m = 20
output_train = np.load(f'train_data/{data}_train.npy')[:, ::m]

input_test = np.load('test_data/laser_parameters_test.npy')
results = []
trans_results = model_analysis('lc', data, input_train, output_train)
results_transmittance = trans_results.get_results()
results.append(results_transmittance)
trans_results.plot_results(results)

data = 'reflectance'

input_train = np.load('train_data/laser_parameters_train.npy')
m = 20
output_train = np.load(f'train_data/{data}_train.npy')[:, ::m]



refl_results = model_analysis('lc', data, input_train, output_train)
results_reflectance = refl_results.get_results()
results.append(results_reflectance)
refl_results.plot_results(results)

            
data = 'transmittance'
input_train = np.load('train_data/laser_parameters_train.npy')
m = 20
output_train = np.load(f'train_data/{data}_train.npy')[:, ::m]


results = []

trans_results = model_analysis('hist', data, input_train, output_train, fraction=1)#, model='lr')
results_transmittance = trans_results.get_results()

results.append(results_transmittance)

trans_results.plot_results(results)    


data = 'reflectance'

input_train = np.load('train_data/laser_parameters_train.npy')
m = 20
output_train = np.load(f'train_data/{data}_train.npy')[:, ::m]
results = []


refl_results = model_analysis('hist', data, input_train, output_train, fraction=1)#, model='lr')
results_reflectance = refl_results.get_results()

results.append(results_reflectance)

refl_results.plot_results(results)