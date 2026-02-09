# Inverse design of silicon-based photonic surfaces with AI Laser Parameter Search and XGBoost
Code and data for the manuscript "Inverse design of silicon-based photonic surfaces via femtosecond laser processing and extreme gradient boosting machine learning". Available on Arxiv:

_______
**DATA DESCRIPTION**
_______
The dataset (in .npy format) used to train and test the models can be found in the train_data and test_data folders, respectively. Combined, the total number of samples are 756. For exmaple, in the the test_data folder we have:

**laser_parameters_test.npy** - the input test data that contains laser processing parameters (speed, spacing) (2D)

**reflectance_test.npy** - the output test data that contains reflectance curves. Reflectance values over 2852 wavelength values.

**transmittance_test.npy** - the output test data that contains transmittance curves. Transmittance values over 2852 wavelength values.

Moreover, the results will be saved in inverse_design_results for running the ALPS+XGBoost combined model (**inverse_design_results.py**), and model_results for analyzing either Linear Regression or XGBoost (**kfold_analysis.py**).

_______
**MODEL ANALYSIS CODE DESCRIPTION**
_______

The model analysis code is contained in **kfold_analysis.py**. This code can output either the learning curve (RMSE as a function of the training set) or the RMSE histogram analysis. The following code can be used to obtain either, based on the transmittance or reflectance data. All imports are in the code.

```python

data = 'transmittance' #we define the data type
input_train = np.load('train_data/laser_parameters_train.npy') #input data
m = 10 #we only select 286 wavelength values since
output_train = np.load(f'train_data/{data}_train.npy')[:, ::m] #output data

results = []
trans_results = model_analysis('lc', data, input_train, output_train/100) #set 'lc' for learning curve analysis
results_transmittance = trans_results.get_results() #get the results
results.append(results_transmittance)
trans_results.plot_results(results) #plot results helper function


data = 'reflectance'

input_train = np.load('train_data/laser_parameters_train.npy')
m = 10
output_train = np.load(f'train_data/{data}_train.npy')[:, ::m]
results = []

refl_results = model_analysis('hist', data, input_train, output_train/100, fraction=1, model='lr') #in this example we have 'hist' for RMSE histogram analysis, and model='lr' for linear regression, if we set 'xgb' or no argument, it defaults to XGBoost regression
results_reflectance = refl_results.get_results()

results.append(results_reflectance)

refl_results.plot_results(results)
```

_______
**ALPS OPTIMIZER**
_______

The ALPS Optimizer source code is contained in the ALPSOptimizer folder. Setting up the optimizer and running it can be achieved through the following example:


```python
import sys
sys.path.insert(0, 'ALPSOptimizer/src')
from ALPS import ALPS #import ALPS based on this relative path

evals = 250 #number of evaluations of the objective function
repeats = 1 #if we want multiple repeats
m=10

def f(x): #the black box function that returns the evaluated design value
    return evaluation(x)

target = [1, 1] #target f value we are interested in, can be vector or scalar value


test_full = ALPS(target=target,
            max_evals=evals,
            initial_sample_size=20, #number of initial samples
            model_batch_size=10, #number of smaples we evaluate per iteration
            lb=lb, #lower boundary of the search space
            ub=ub, #upper boundary of the search space
            experiment=f)

results_alps = test_full.run(repeats) #returns the convergence curve

```


_______
**INVERSE DESIGN ANALYSIS CODE DESCRIPTION**
_______

The inverse design model analysis code is contained in **inverse_design_analysis.py**. This trains a forward XGBoost model, loads the ALPS Optimizer and based on a target absorptance curve finds the right design paramaters (scanning speed and spacing). In this code, the target absorptance is hard coded as a 100% curve. All imports can be found in the code.

```python

k = 10 #we select 286 wavelength values
laser_parameters_train = np.load('train_data/laser_parameters_train.npy')
reflectance_train = np.load('train_data/reflectance_train.npy')[:, ::k]/100
transmittance_train = np.load('train_data/transmittance_train.npy')[:, ::k]/100

laser_parameters_test = np.load('test_data/laser_parameters_test.npy')
reflectance_test = np.load('test_data/reflectance_test.npy')[:, ::k]/100
transmittance_test = np.load('test_data/transmittance_test.npy')[:, ::k]/100

wavelengths = np.load('data/wavelengths.npy')[::k]

#we load the train and test data here for the xgboost model to train on the fly and test the accuracy
train_data_r = (laser_parameters_train, reflectance_train)
train_data_t = (laser_parameters_train, transmittance_train)

test_data_r = (laser_parameters_test, reflectance_test)
test_data_t = (laser_parameters_test, transmittance_test)


p=1 #fraction of train dataset used, i.e. 1 = 100%
n=5 #top n predictions returned from ALPS that matech the target

test = inverseDesignAnalysis(train_data_r, test_data_r,
                             train_data_t, test_data_t, p, n,
                             wavelengths=wavelengths)

test.get_plot_predictions() #plot the predictions helper function
```






