import matplotlib.pyplot as plt
import numpy as np

from matplotlib.markers import MarkerStyle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import os


import sys
sys.path.insert(0, '../models_benchmarks')

import benchmarks_functions as bf



plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})

class postprocess_dict:
    
    def __init__(self, results_):
        
        self.results_ = results_
    
    def get_dict(self):
        
        full_res_dict = {'evals': None, 'fitness': [] , 'n_runs': None,
                          'optimizer': None, 'solutions': [] }

        full_res_dict['evals'] = self.results_[0]['evals']
        full_res_dict['n_runs'] = np.sum([res['n_runs'] for res in self.results_])
        full_res_dict['optimizer'] = self.results_[0]['optimizer']
        full_res_dict['fitness'] = np.concatenate([res['fitness'] for res in self.results_])
        full_res_dict['solutions'] = np.concatenate([res['solutions'] for res in self.results_])

        return full_res_dict
    

class save_results:
    
    def __init__(self, results, results_folder):
        
        self.results = results
        self.results_folder = results_folder

    @staticmethod
    def sort_fitness(unsorted):
        
      
        new_list = [unsorted[0]]
        indx = [0]
        
        for i in range(len(unsorted)):
            
            if unsorted[i] < new_list[-1]:
                new_list.append(unsorted[i])
                indx.append(i)
            else:
                new_list.append(new_list[-1])
                indx.append(indx[-1])
        
        return new_list
    
    @staticmethod
    def get_index(unsorted):
        
      
        new_list = [unsorted[0]]
        indx = [0]
        
        for i in range(len(unsorted)):
            
            if unsorted[i] < new_list[-1]:
                new_list.append(unsorted[i])
                indx.append(i)
            else:
                new_list.append(new_list[-1])
                indx.append(indx[-1])

        return indx[-1]
    


    def savetxt(self, model_size):
        
        n_runs = self.results[0]['n_runs']
        evals_ = self.results[0]['evals']

        results_folder = f'{self.results_folder}/results_data_{model_size}'

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        print (results_folder)
        run_statistics = []
        for dict_ in self.results:
            
            candidates = dict_['solutions']
            fitness_unsorted = dict_['fitness']
            optimizer = dict_['optimizer']


            fitness = [self.sort_fitness(i) for i in fitness_unsorted]
            best_indx = [self.get_index(i) for i in fitness_unsorted]

            best_candidates = [np.array(candidates)[i, best_indx[i]] for i in range(len(candidates))]
            mean_fitness = np.mean(fitness, axis=0)
            max_fitness = np.max(fitness, axis=0)
            min_fitness = np.min(fitness, axis=0)
            std_fitness = np.std(fitness, axis=0)
            rnd = np.random.randint(1,2000)
            # np.savetxt(f'{results_folder}/ALPS_{rnd}_{evals_}_{n_runs}_fitness_{model_size}.txt', fitness)
            #np.savetxt(f'{results_folder}/{optimizer}_{evals_}_{n_runs}_meanfitness_{model_size}.txt', mean_fitness)
            #np.savetxt(f'{results_folder}/{optimizer}_{evals_}_{n_runs}_minfitness_{model_size}.txt', min_fitness)
            #np.savetxt(f'{results_folder}/{optimizer}_{evals_}_{n_runs}_maxfitness_{model_size}.txt', max_fitness)
            np.savetxt(f'{results_folder}/{optimizer}_{rnd}_{evals_}_{n_runs}_bestcandidates_{model_size}.txt', best_candidates)
            
            run_statistics.append([str(optimizer), mean_fitness[-1], max_fitness[-1], min_fitness[-1], std_fitness[-1]])
        
        # w = open(f'{results_folder}/statistics_evals_{evals_}_runs_{n_runs}_{model_size}.txt', 'w')
        # for i in range(len(run_statistics)):
        #     w.write(str(run_statistics[i][0]) + ' ' + str(run_statistics[i][1]) + ' ' +  str(run_statistics[i][2]) + ' ' +  str(run_statistics[i][3]) + ' ' +  str(run_statistics[i][4]) + '\n')
            
        # w.close()
        
        # np.savetxt(f'{results_folder}/statistics_evals_{evals_}_runs_{n_runs}.txt', run_statistics)
            
class plot_results:
    
    def __init__(self, results, results_folder):
        
        self.results = results
        self.results_folder = results_folder


    @staticmethod
    def sort_fitness(unsorted):
        
      
        new_list = [unsorted[0]]
        indx = [0]
        
        for i in range(len(unsorted)):
            
            if unsorted[i] < new_list[-1]:
                new_list.append(unsorted[i])
                indx.append(i)
            else:
                new_list.append(new_list[-1])
                indx.append(indx[-1])
        
        return new_list
    
    @staticmethod
    def get_index(unsorted):
        
      
        new_list = [unsorted[0]]
        indx = [0]
        
        for i in range(len(unsorted)):
            
            if unsorted[i] < new_list[-1]:
                new_list.append(unsorted[i])
                indx.append(i)
            else:
                new_list.append(new_list[-1])
                indx.append(indx[-1])

        return indx[-1]
    
        
    def compare_metrics(self, function, log_on, std_show, return_mean_fitness=False, title_string=None):
        
        
        # colors = ['red', 'blue', 'orange', 'green', 'cyan', 'black', 'purple', 'gold', 'pink']
        colors = ['red', 'blue', 'orange', 'green', 'cyan', 'black', 'purple', 'gold', 'pink',
                  'magenta', 'lime', 'gray', 'maroon', 'navy', 'olive', 'teal', 'aqua', 'silver', 'chocolate' ]

        c = 0
               
        evals = np.arange(1, self.results[0]['evals'], 1)
        function = function
        
        plt.figure(figsize=(6,5))
        
        n_runs = self.results[0]['n_runs']
        optimizer = self.results[0]['optimizer']
        evals_ = self.results[0]['evals']

        
        # plt.title(f'{n_runs} runs, {function}', fontsize=15)

        for dict_ in self.results:
            

                evals_ = dict_['evals']
                evals = np.arange(1, evals_+1, 1)
                candidates = dict_['solutions']
                fitness_unsorted = dict_['fitness']

                fitness = [self.sort_fitness(i) for i in fitness_unsorted]
                best_indx = [self.get_index(i) for i in fitness_unsorted]

                best_candidates = [np.array(candidates)[i, best_indx[i]] for i in range(len(candidates))]

                mean_fitness = np.mean(fitness, axis=0)#[:len(fitness[0])]#*100
                
                if std_show == True:
                    std_fitness = np.std(fitness, axis=0)#[:len(fitness[0])]#*100
                    min_fitness = np.min(fitness, axis=0)
                    max_fitness = np.max(fitness, axis=0)
                    perc10 = np.percentile(fitness, 10, axis=0)
                    perc90 = np.percentile(fitness, 90, axis=0)

                
                if log_on == True:
                    mean_fitness = np.log10(mean_fitness)
                    std_fitness = np.log10(std_fitness)
                    perc10 = np.log10(perc10)
                    perc90 = np.log10(perc90)
                
                print (f"best f {dict_['optimizer']}", mean_fitness[-1], len(fitness[0]), 'evals')
                
                plt.plot(np.arange(0, len(mean_fitness), 1), mean_fitness, color=colors[c],alpha=1, linewidth=3, label = f"{dict_['optimizer']}")
                

                if std_show == True:
                    plt.fill_between(np.arange(0, len(mean_fitness), 1), mean_fitness, perc10, color=colors[c],alpha=0.025)
                    plt.fill_between(np.arange(0, len(mean_fitness), 1), mean_fitness, perc90, color=colors[c],alpha=0.025)
                    plt.plot(np.arange(0, len(mean_fitness), 1), perc10, linewidth=0.5, color=colors[c],alpha=1)
                    plt.plot(np.arange(0, len(mean_fitness), 1), perc90, linewidth=0.5, color=colors[c],alpha=1)
                    
                if log_on == True:
                    plt.ylabel('log($\epsilon$)')
                    
                else:
                    plt.ylabel('$\epsilon$')
                    
                # if 'emissivity' in function:
   
                #     plt.ylim(-0.05, 1.05)
                
                if 'logistic_growth' in function and log_on is False:
                    plt.ylim(0, 300)
                
                if 'oscillator' in function and log_on is False:
                    plt.ylim(0.05, 1)

                # elif 'oscillator' in function:
                #     plt.xlabel('Time (s)')
                #     plt.ylabel('Amplitude') 
                    
                plt.xlabel('Experiments')
                plt.legend(fontsize=12)
                c+=1
                
                ax = plt.gca()

                for axis in ['top', 'bottom', 'left', 'right']:
                  ax.spines[axis].set_linewidth(2)
                
                plt.tight_layout()
                if title_string != None:
                    plt.title(title_string, fontsize=12)
                plt.savefig(f'{self.results_folder}/{function}_{n_runs}_convergence_rate_std_{std_show}.pdf', dpi=400)   
        
        if return_mean_fitness == True:
        
            return mean_fitness[-1]
        
    def plot_solution(self, fname, target, domain, alpha=0.05, title_string=None):

        colors = ['red', 'blue', 'orange', 'green', 'cyan', 'black', 'purple', 'gold', 'pink']
        c = 0
               
        evals = np.arange(1, self.results[0]['evals'], 1)
        
        plt.figure(figsize=(6,5))
        
        n_runs = self.results[0]['n_runs']
        optimizer = self.results[0]['optimizer']
        evals_ = self.results[0]['evals']

        
        # plt.title(f'{n_runs} runs, {fname}', fontsize=15)
        
        plt.plot(domain, target*100, color='black', linewidth=3, linestyle='--', label='Target')
        
        
        for dict_ in self.results:
            
                candidates = dict_['solutions']
                fitness_unsorted = dict_['fitness']

                fitness = [self.sort_fitness(i) for i in fitness_unsorted]

                best_indx = [self.get_index(i) for i in fitness_unsorted]
                best_candidates = np.array([np.array(candidates)[i, best_indx[i]] for i in range(len(candidates))])
                values = []
                for i in best_candidates:
                    val = bf.function(fname, target, model_size=1).evaluate(i)
                    values.append(val[1][0])
                    
                    
                # print (values)
                
                # for i in values:
                #     plt.plot(domain, i, color=colors[c], alpha=1)
                
                
                mean_fitness = np.mean(fitness, axis=0)#[:len(fitness[0])]#*100
                
                plt.xlabel(r'Wavelenght ($\upmu$m)')
                plt.ylabel('Absorbance (\%)')
                plt.ylim(0, 105)
                
                scale=100
                    
                plt.plot(domain, np.mean(values, axis=0)*scale, color=colors[c], label = "ALPS")
                # plt.fill_between(domain, np.mean(values, axis=0), np.mean(values, axis=0) - np.std(values, axis=0), color=colors[c], alpha=alpha)
                # plt.fill_between(domain, np.mean(values, axis=0), np.mean(values, axis=0) + np.std(values, axis=0), color=colors[c], alpha=alpha)
                plt.fill_between(domain, np.mean(values, axis=0)*scale, np.percentile(values, 90, axis=0)*scale, color=colors[c], alpha=alpha)
                plt.fill_between(domain, np.mean(values, axis=0)*scale, np.percentile(values, 10, axis=0)*scale, color=colors[c], alpha=alpha)
                # plt.plot(domain, np.mean(values, axis=0) - np.std(values, axis=0), linewidth=0.5, color=colors[c])
                # plt.plot(domain, np.mean(values, axis=0) + np.std(values, axis=0), linewidth=0.5, color=colors[c])

                ax = plt.gca()
 
                for axis in ['top', 'bottom', 'left', 'right']:
                  ax.spines[axis].set_linewidth(2)
                
                plt.tight_layout()

                plt.legend(fontsize=10)#, loc='lower right')
                c+=1
                
                              
        if title_string != None:
            plt.title(title_string, fontsize=12)

        plt.savefig(f'{self.results_folder}/{fname}_{n_runs}_solutions.pdf', dpi=400)   

            
    def plot_individual_solutions(self, fname, target, domain, alpha=0.05):
    
        colors = ['red', 'blue', 'orange', 'green', 'cyan', 'black', 'purple', 'gold', 'pink']
        c = 0
               
        evals = np.arange(1, self.results[0]['evals'], 1)
                
        n_runs = self.results[0]['n_runs']
        optimizer = self.results[0]['optimizer']
        evals_ = self.results[0]['evals']
    
               
        
        
        for dict_ in self.results:
                plt.figure(figsize=(6,5))
                candidates = dict_['solutions']
                fitness_unsorted = dict_['fitness']
    
                fitness = [self.sort_fitness(i) for i in fitness_unsorted]
    
                best_indx = [self.get_index(i) for i in fitness_unsorted]
                best_candidates = np.array([np.array(candidates)[i, best_indx[i]] for i in range(len(candidates))])
                values = []
                for i in best_candidates:
                    val = bf.function(fname, target, model_size=1, return_value=True).evaluate(i)
                    values.append(val[1][0])
                    
                    
       
      
                
                
                mean_fitness = np.mean(fitness, axis=0)#[:len(fitness[0])]#*100
                
                plt.xlabel(r'Wavelenght ($\upmu$m)')
                plt.ylabel('Absorbance (\%)')
                plt.ylim(0, 105)
                        
                scale = 100
                
                plt.plot(domain, target*scale, color='black', linewidth=3, linestyle='--', label='Target')
                plt.plot(domain, np.mean(values, axis=0)*scale, color=colors[c], label = "ALPS")
                # plt.fill_between(domain, np.mean(values, axis=0), np.mean(values, axis=0) - np.std(values, axis=0), color=colors[c], alpha=alpha)
                # plt.fill_between(domain, np.mean(values, axis=0), np.mean(values, axis=0) + np.std(values, axis=0), color=colors[c], alpha=alpha)
                plt.fill_between(domain, np.mean(values, axis=0)*scale, np.percentile(values, 90, axis=0)*scale, color=colors[c], alpha=alpha)
                plt.fill_between(domain, np.mean(values, axis=0)*scale, np.percentile(values, 10, axis=0)*scale, color=colors[c], alpha=alpha)
                # plt.plot(domain, np.mean(values, axis=0) - np.std(values, axis=0), linewidth=0.5, color=colors[c])
                # plt.plot(domain, np.mean(values, axis=0) + np.std(values, axis=0), linewidth=0.5, color=colors[c])
    
                ax = plt.gca()
    
                for axis in ['top', 'bottom', 'left', 'right']:
                  ax.spines[axis].set_linewidth(2)
                
                plt.tight_layout()
    
                plt.legend(fontsize=18)
                c+=1

                              
        
                plt.savefig(f"{self.results_folder}/{fname}_{n_runs}_solutions_{dict_['optimizer']}.pdf", dpi=400)   

        

            
        
        
        
