import sys
import inspect
import os
import numpy as np
import json
from datetime import date
from random import choices,random
from pathlib import Path
from itertools import product,permutations

import learning_moves
import experiment_classes
import config_file
import uniform_search

class learning_particle:
    '''
    This class contains the rjMCMC learning agent outlined in 
    (TODO reference publication). The class is modular calling in from 
    "config_file.py", "experiment_classes.py", "learning_moves".py and 
    "uniform_search.py" when run for the first time for a new Hilbert Space.

    The class takes as intial information:

    "experiment_list" : ["experiment1", "experiment2"]
        The strings passed here should match the names of the functions defined 
        in "learning_moves.py".

    "data_directory_list" : ["dir1/file.npy", "dir2/file.npy"]
        These strings point to directories that contain [x,y] data for the 
        experiment with the matching index above. 

        Therefor the function "experiment1" in "learning_moves.py" should, given 
        the correct model give the data collected in "dir1/file.py" which holds 
        for as many experiments as one wishes. Experiment functions can be used
        multiple times for different experiments, however this has limited 
        use-cases.

    "initial_particle" : 'random' | {...}
        This variable accepts the string 'random' or a dictionary denoting a 
        model. In the former case the agent will construct a random model 
        available with the number of the operators of Hamiltonian and 
        Lindbladian flavour informed by the "config_file.py" file.

        In the case that a dictionary is passed the agent will any 
        "immutable_processes if defined in "config_file.py" and missing from the 
        model, if not missing then the agent will accept the given model
        dictionary and move on.
    
    "number_of_sites" : int    
        This controls the Hilbert space of the learning and is the number of 
        concatinated 2 level emitters the system should consider. The algorithm 
        scales exponentially with this value and no value larger than 3 has been
        used so far in the developement of the algorithm.

    "verbose" : bool
        This is a debugging tool and returns additional errors and information 
        during learning from all scripts and functions.
        
    '''
    def __init__(self,  experiment_list,                data_directory_list,
                        initial_particle = 'random',    number_of_sites = 2,
                        verbose = False):
        #####
        # Parsing inputs
        #####
        self.path = os.getcwd()
        self.number_of_sites = number_of_sites
        self.verbose = verbose
        self.initial_particle = initial_particle
        self.learning_params = config_file.learning_params.copy()
        


        self.check_file_integrity()                                             # Function for intialisation checks
        self.import_operators()                                                 # Initialises operators in learning
                                                                                # Populated uniform transformation options
        uniform_search.uniform_search(dim = self.number_of_sites,verbose=self.verbose)                                  

        #####
        # Imports experiments from "experiment_classes.py"
        #####
        self.experiments = []
        for e,d in zip(experiment_list,data_directory_list):
            exp_func_ = getattr(experiment_classes,e)
            self.experiments.append(exp_func_(self.number_of_sites,self.operators,d))
            if verbose:
                print("Experiment Loaded Correctly: \t\t" + e)
        
        #####
        # Initialises Learning moves
        #####
        self.accepted = 0
        self.move_weights = []
        self.MCMC_moves = {}

        #####
        # Parses Learning Moves
        #####
        for m in inspect.getmembers(learning_moves, inspect.isclass):           # Iterates through "learning_moves.py"
            if m[0] != 'ABC' and m[0] != 'Move' and m[0] != 'permutations':     # Ignores exceptions
                self.move_weights.append(self.learning_params[m[0]+'_chance'])
                print(self.verbose)
                self.MCMC_moves[m[0]] = m[1](self.learning_params,[self.H_ops,self.L_ops],verbose = self.verbose)
        # Normalises move probabilities
        self.move_weights = list(np.asarray(self.move_weights)/sum(self.move_weights))

        # Saves updated normalised move probabilities
        for n,k in enumerate(self.MCMC_moves.keys()):                           
            self.learning_params[k+'_chance'] = self.move_weights[n]

        #####
        # Initialises first particle
        #####
        if initial_particle == 'random':
            # Randomly constructs particle from "config_file.py" preferences

            # Identifies moves to add processes
            adding_indexes = [n for n,k in enumerate(list(self.MCMC_moves.keys())) if k.startswith('ADDING')]
            adding_keys = [k for k in list(self.MCMC_moves.keys()) if k.startswith('ADDING')]

            # temporarily normalises adding moves to failry construct particles
            temp_weights = [self.move_weights[i] for i in adding_indexes]

            # initialises empty particles
            self.particle = {}

            # adds immutable processes
            for ind,pro in enumerate(self.learning_params['immutable_processes']):
                if self.learning_params['immutable_rates'][ind] == None:
                    self.particle[pro] = np.random.gamma(self.learning_params['parameter_prior_shape'],scale = self.learning_params['parameter_prior_scale'])
                else:
                    self.particle[pro] = self.learning_params['immutable_rates'][ind]

            # parses from "config_file.py"
            expected_parameters = int(self.learning_params['lindbladian_exponential_rate']+self.learning_params['hamiltonian_exponential_rate'])

            # iteratively adds processes with moves that add.
            while len(self.particle.keys()) <= expected_parameters:
                move = choices(adding_keys,weights=temp_weights)[0]
                temp = self.MCMC_moves[move].Enact([self.H_ops,self.L_ops],self.particle)
                if temp != None:
                    self.particle = temp
                elif verbose:
                    print("Failed Move in Model Randomisation: \t" + move,temp)

        elif 'bkg' not in initial_particle.keys():
            # Adds background processes when necessary 
            if self.verbose:
                print('Background count rate not present in preset model and will be added.')
            initial_particle['bkg'] = np.random.gamma(self.learning_params['parameter_prior_shape'],scale = self.learning_params['parameter_prior_scale'])/10
            self.particle = initial_particle.copy()
        else:
            # Accepts particle
            self.particle = initial_particle.copy()

        # Calculates results, beta and posterior for the first particle and initialises data structure.
        self.current_exp_results = [e.Calculation(self.particle,step=0) for e in self.experiments]
        self.current_exp_weights = [e.Experiment_Weighting(self.particle, step = 0) for e in self.experiments]
        self.current_exp_diffs = [np.abs(res-exp.sampled_data[1]) for res,exp in zip(self.current_exp_results,self.experiments)]
        self.current_exp_posterior = [-1*W/2*np.dot(d.T,np.multiply((e.chosen_weights),d)) for W,d,e in zip(self.current_exp_weights,self.current_exp_diffs,self.experiments)]

        # Initialises data structure for learning monitoring
        self.learning_data = {'step number':[0],'particle':[self.particle.copy()],'posterior':[sum([self.current_exp_posterior][0])],'acceptance rate':[1.0],'log likelihood':[0],'experiment weighting':[self.current_exp_weights]}
        self.learning_data['proposed move dict'] = {k:[] for k in self.MCMC_moves.keys()}
        self.learning_data['accepted move dict'] = {k:[] for k in self.MCMC_moves.keys()}
        print('INITIALISED')
    def check_file_integrity(self):
        '''
        For the proper fucntioning of the agent multiple strings act as 
        pointers. "config_file.py" defines rates for functions in 
        "learning_moves.py", "experiment_list" as a variable input for the 
        "learning_agent" makes reference to experiments in 
        "experiment_classes.py".

        This function checks that these pointers are all correct before the 
        learning begins as the default error messages are not adequate to debug 
        these issues.
        '''
        #####
        # Checks learning moves are correct in "config_file.py"
        #####
        chance_parameters = [k[:-7] for k in self.learning_params.keys() if k[-7:] == '_chance']
        print(inspect.getmembers(learning_moves, inspect.isclass))
        move_functions = len(inspect.getmembers(learning_moves, inspect.isclass)) - 3                                   # The 3 is ABC, Move, and permutations

        #####
        # Checks that the same number of moves are expects in script and file
        #####
        if len(chance_parameters) != move_functions:                                                                    # Errors when more or less weights are provided for moves that don't exist.
            sys.exit("From config_file.py there are " + str(len(chance_parameters)) +
                    " '_chance' parameters, and from learning_moves_py there are " +
                    str(move_functions) + " functions, this needs to match. Please" +
                    " remember to match names of functions to config chances.")
            
        #####
        # Checks that the naming of probabilities are correct.
        #####
        missed = 0 
        for c in inspect.getmembers(learning_moves, inspect.isclass):
            if c[0] in chance_parameters:
                if self.verbose:
                    print("Learning Move Loaded Correctly:\t\t" + c[0])
            else:
                missed+=1
        if missed > 3:
            sys.exit("Name missmatch between config_file.py and learning_moves.py")                                 

    def import_operators(self):
        '''
        The library of usable operators is the backbone of the learning agent, 
        it defines the learning space. It identifies allowed processes and this 
        function preforms the construction of this space. The function first 
        checks that the files do not exist in the directory to begin with and 
        if the file is missing the function does the work to construct it. The 
        file is unique tothe hilbert sapce and the variable "complexity" which 
        is explained in the publication (TODO link publication).

        In the case that the operators have not already been found then the 
        agent uses the "core_operator_dict" to construct the library. This can 
        be edited by a user but the user should bewaryof a few things.

        1) Each key must be a single unique character.
        2) Each matrix should be the same dimension.
        3) This notation should be understandable to at least the one to run the
            code as it is the langauage in whihc all models are expressed.
        '''
        
        op_path = (self.path  + r"/opertators_"+str(self.number_of_sites)+      # Identifies expected location of library
                                r"_complexity_"+str(self.learning_params['complexity'])+
                                r".npy")
        self.learning_params['operator path'] = op_path                         # Saves location for debugging and transparency
        if os.path.isfile(op_path):
            
            #####
            # Loads operator library
            #####
            if self.verbose:
                print("Operator Dictionary Loaded From: \t" + op_path)
                                
            numpy_ops = np.load(op_path, allow_pickle=True)                     # loads if available
            
            self.operators = {k:v for k,v in zip(list(numpy_ops[0,:]),list(numpy_ops[1,:]))}
        else:
            
            #####
            # Constructs library
            #####
            def _lindblad_from_string(mtx_key,core_operator_dict):
                '''
                constructs lindblad matrices from strings. "hold" allows for the 
                comprehension of "+".

                Finally we only allow operators with 1s or 0s in the real or 
                imaginary parts so any value is set to 0 if 0 or 1 if not. This 
                is then used to identify unique operators, otherwise 
                x != x+x+x+x+... when functionally they are the same operator.
                '''
                hold = np.zeros((2**self.number_of_sites,2**self.number_of_sites))
                mtx = 1
                for c in mtx_key:
                    if c == '+':
                        hold = mtx
                        # 1 to allow tensor product to work properly
                        mtx = 1
                    else:
                        mtx = np.kron(mtx,core_operator_dict[c])
                mtx = mtx + hold

                # Final set to 1 or 0
                return np.where(np.real(mtx) == 0, 0, np.sign(np.real(mtx))) + 1j*np.where(np.imag(mtx) == 0, 0, np.sign(np.imag(mtx)))
            core_operator_dict = {
            'u': np.array([                                                     # up
                [1 + 0.j, 0 + 0.j],
                [0 + 0.j, 0 + 0.j]
            ]),
            'd': np.array([                                                     # down
                [0 + 0.j, 0 + 0.j],
                [0 + 0.j, 1 + 0.j]
            ]),
            'a': np.array([                                                     # Add
                [0 + 0.j, 1 + 0.j],
                [0 + 0.j, 0 + 0.j]
            ]),
            's': np.array([                                                     # Subtract
                [0 + 0.j, 0 + 0.j],
                [1 + 0.j, 0 + 0.j]
            ])}

            '''
            ,
            'U': np.array([                                                     # up positive complex
                [0 + 1.j, 0 + 0.j],
                [0 + 0.j, 0 + 0.j]
            ]),
            'D': np.array([                                                     # down positive complex
                [0 + 0.j, 0 + 0.j],
                [0 + 0.j, 0 + 1.j]
            ]),
            'A': np.array([                                                     # Add positive complex
                [0 + 0.j, 0 + 1.j],
                [0 + 0.j, 0 + 0.j]
            ]),
            'S': np.array([                                                     # Subtract positive complex
                [0 + 0.j, 0 + 0.j],
                [0 + 1.j, 0 + 0.j]
            ]),
            'W': np.array([                                                     # up negative complex
                [0 - 1.j, 0 + 0.j],
                [0 + 0.j, 0 + 0.j]
            ]),
            'B': np.array([                                                     # down negative complex
                [0 + 0.j, 0 + 0.j],
                [0 + 0.j, 0 - 1.j]
            ]),
            'Z': np.array([                                                     # Add negative complex
                [0 + 0.j, 0 - 1.j],
                [0 + 0.j, 0 + 0.j]
            ]),
            'T': np.array([                                                     # Subtract negative complex
                [0 + 0.j, 0 + 0.j],
                [0 - 1.j, 0 + 0.j]
            ])
            '''
            if self.verbose:
                print("Building Operator Dictionary To: \t" + op_path)
                print("Operators Constructed With \t\t"+ str(list(core_operator_dict.keys())))

            self.operators = {}
            simple_terms = ["".join(s) for s in product(list(core_operator_dict.keys()),repeat = self.number_of_sites)]
            #####
            # Remove two photon processes if wanted
            #####
            if self.number_of_sites ==2:
                for i in ['a','s','A','Z','S','T']:
                    for j in ['a','s','A','Z','S','T']:
                        try:
                            simple_terms.remove(i+j)
                        except:
                            pass
                print(simple_terms)

            # constructs ALL postible operators.
            string_terms = simple_terms + ["+".join(s) for s in permutations(simple_terms,self.learning_params['complexity'])]

            # extracts only unique operators
            for n,k in enumerate(string_terms):
                mtx = _lindblad_from_string(k,core_operator_dict)
                if len([mtx for m in list(self.operators.values()) if np.allclose(mtx,m)]) == 0:
                    self.operators[k] = mtx

                if self.verbose and n%50 == 0 and n != 0:
                    print("Operators Explored: \t\t\t" + str(100*n/len(string_terms)) + r"%")
            
            self.operators = {k:v for k,v in zip(list(self.operators.keys()),list(self.operators.values())) if (np.isreal(np.matrix(v)).all() or np.allclose(np.matrix(v).H, np.matrix(v))) and np.any(v)}
            if self.verbose:
                print("Unique Lindbladians Found: \t\t" + str(len(self.operators)))
            np.save(op_path, np.array([list(self.operators.keys()),list(self.operators.values())], dtype=object), allow_pickle=True)

        # Parses Lindblad subset and Hamiltonian subset.
        self.L_ops = {k:v for k,v in zip(list(self.operators.keys()),list(self.operators.values())) if np.isreal(np.matrix(v)).all() and np.all(v>=0) and np.any(v)}
        self.H_ops = {k:v for k,v in zip(list(self.operators.keys()),list(self.operators.values())) if np.allclose(np.matrix(v).H, np.matrix(v)) and np.any(v)}

        print('Hamiltonian Terms: ',self.H_ops)
        print('Lindbladian Terms: ',self.L_ops)
            
    def learning_loop(self):
        '''
        This is the main function that does the learning. It can be called 
        multiple times if needed.
        '''

        # Saves parameters for the learn
        Path(self.learning_params['project_name']).mkdir(parents=True,exist_ok=True) 
        with open(self.learning_params['project_name']+'/data_' + str(date.today()) + '_' + 'config.json','w') as f:
            json.dump(self.learning_params,f)

        for s in range(1,self.learning_params['chain_steps']):                  # Iterates over through the learning
            
            #####
            # makes trial particle
            #####
            trial_particle = None
            while trial_particle == None:
                move = choices(list(self.MCMC_moves.keys()),weights=self.move_weights)[0]
                trial_particle = self.MCMC_moves[move].Enact([self.H_ops,self.L_ops],self.particle)

            # Calculates posterior for trial particle
            trial_exp_results = [e.Calculation(trial_particle,step = s) for e in self.experiments]
            trial_exp_diffs = [np.abs(res-exp.sampled_data[1]) for res,exp in zip(trial_exp_results,self.experiments)]
            trial_exp_posterior = [-1*W/2*np.dot(d.T,np.multiply((e.chosen_weights),d)) for W,d,e in zip(self.current_exp_weights,trial_exp_diffs,self.experiments)]

            # subsampling resampling when necessary (not used in publication)
            if len(self.current_exp_diffs[0]) != len(self.experiments[0].sampled_data[1]):
                self.current_exp_results = [e.Calculation(self.particle,step=s) for e in self.experiments]
                self.current_exp_diffs = [np.abs(res-exp.sampled_data[1]) for res,exp in zip(self.current_exp_results,self.experiments)]
                self.current_exp_posterior = [-1*W/2*np.dot(d.T,np.multiply((e.chosen_weights),d)) for W,d,e in zip(self.current_exp_weights,self.current_exp_diffs,self.experiments)]

            # calculates accept reject ratio
            AR,log_likelihood = self.MCMC_moves[move].Accept_Reject(self.particle,np.sum(trial_exp_posterior),np.sum(self.current_exp_posterior),self.operators)

            # runs MCMC accept reject
            accepted = False
            N = random()
            if N <= AR:

                # accepts particle
                self.accepted += 1
                accepted = True
                print('##########################################' , '\n'
                'Accepted Particle: ',s,'\n',
                '\tAccptance Rate: \t\t\t' ,AR,'\n',
                '\tTrial and Current Probabilities: \t', np.sum(trial_exp_posterior),np.sum(self.current_exp_posterior),'\n',
                "\tModel's Terms and Values: \t\t",trial_particle,'\n',
                "\tUpdate: \t\t\t\t",move, '\n'+
                '##########################################')
                self.current_exp_results = trial_exp_results
                self.current_exp_diffs = trial_exp_diffs
                self.current_exp_posterior = trial_exp_posterior

                self.particle = trial_particle
            else:
                # rejects particle
                if self.verbose:
                    print(
                        '#####','\n',
                        'Rejected Particle: \t',s,'\n',
                        'AR:\t\t',AR,'\n'
                        'Move: \t\t\t',move,'\n'+
                        '#####'
                    )
            self.accept_particle(s,AR,log_likelihood,move,accepted)

    def accept_particle(self,step,AR,log_likelihood,move, accepted):
        '''
        Functionally just sequesters data for analysis in the future        
        '''
        self.learning_data['step number'].append(step)
        self.learning_data['particle'].append(self.particle.copy())
        self.learning_data['posterior'].append(sum([self.current_exp_posterior][0]))
        self.learning_data['acceptance rate'].append(AR)
        self.learning_data['log likelihood'].append(log_likelihood)
        self.learning_data['experiment weighting'].append(self.current_exp_weights)
        self.learning_data['proposed move dict'][move].append(step)
        if accepted:
            self.learning_data['accepted move dict'][move].append(step)
    
    def save(self):
        '''
        Saves "self.learning_data" and particles in a json file (TODO include csv and sql saving options)
        '''
        Path(self.learning_params['project_name']).mkdir(parents=True,exist_ok=True)                                    # Create location for file saving                                                                      
        with open(self.learning_params['project_name']+'/data_' + str(date.today()) + '_' + self.learning_params['project_name'] + str(int(100000000000*np.random.random())) + '.json', 'w') as f:         # Append name with random digit to make unique file
            try:
                json.dump(self.learning_data,f)
            except:
                sys.exit('Save Failed',self.learning_data)


data_name = [r"/both_dot_g2_after_edit.npy",r"/both_dot_lifetime_after_edit.npy"]
root_addess = r"C:\Users\sw2009\Documents\Python_Scripts\SCRIPTS\Cristian's Work"
#root_addess = r"/home/sw2009/python_scripts"


hi1 = learning_particle(['g2','Lifetime'],
    [root_addess + d for d in data_name],number_of_sites=2, verbose=False)


if hi1.verbose:
    print('---Learning Class Intitialised---')
hi1.learning_loop()
if hi1.verbose:
    print('---Learning Complete---')
hi1.save()
if hi1.verbose:
    print('---Save Complete---')


