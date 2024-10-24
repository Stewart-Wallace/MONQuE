from random import random, choice
from math import gamma,comb
from abc import ABC, abstractmethod
from json import load
from os import getcwd
from scipy.stats import gamma as scipy_gamma
import sys
import numpy as np
from itertools import permutations

def conjugate_string(operator:str,operators:dict) -> str:
    '''
    To conjugate a matrix one would preform a mathematical operation to make:

    [[a+ib c+id] -> [[a+ib e-if]
    [e+if g+ih]]    [c-id g+ih]]

    However by representing these matrices in string form this operation is not 
    easily performed mathemtically.

    Here each pair in the available library are hardcoded with a plan to
    generalise in the next update

    Additionally though A+B = B+A in the operation of matrices the representation
    of the matrices in the string shouldn't matter. Due to the way the library 
    is constructed A+B may be present and B+A may not. An additional step is
    taken to ensure that the operator takes the form of the label present in the
    library of terms.
    '''
    out = operator
    for p,q in [('a','s'),('A','T'),('Z','S')]:                                 # Conjugate transpose pairs of operators
        out = out.replace(p,'!')                                                # TODO generalise
        out = out.replace(q,p)
        out = out.replace('!',q)
    
    #####
    # Find correct iteration of ordering
    #####
    if out[1:] in operators.keys():                 
        return out
    else:
        comps = out[1:].split('+')
        for pp in permutations(comps,len(comps)):
            out = 'L'
            for s in pp:
                if len(out) == 1:
                    out = out + s
                else:
                    out = out + '+' + s
            if out[1:] in operators.keys():
                return out
        sys.exit("Conjugate String not present in Operators: "+out)
    
def new_operator(operators,flavour:str,exc:list) -> str:
    '''
    This function takes a list of exemptions and returns a random operator 
    excluding the stated exemptions with a given flavour.
    '''
    exc = [e[1:] for e in list(exc) if e[0] == flavour]
    possible_strings = [s for s in list(operators.keys()) if s not in exc]      
    if flavour == 'H':
        op_string = choice([s for s in possible_strings])
    if flavour == 'L':
        op_string = choice(list(possible_strings))
    return flavour+op_string

class Move(ABC):
    '''
    Move is the parent class for any move within model space. The initialisation
    parses a probability of the move taking place, for use in the likelihood and
    the current particle that is being changed. These objects should take the 
    below structure:
        1) change: a value between 0 and 1, corresponding to the likelihood of
            this move's selection.
        2) current_particle: A dictionary of string terms as keys and float
            values as values, corresponding to the rates of those terms.
    '''
    def __init__(self,learning_params,operators, verbose = False):
        self.learning_params = learning_params
        self.verbose = verbose
        self.all_hamiltonian_terms = len(list(operators[0].keys()))
        self.all_lindblad_terms = len(list(operators[1].keys()))
        self.all_complex_terms = len([o for o in list(operators[1].keys()) if '+' in o])
        self.g = scipy_gamma(a=self.learning_params['parameter_prior_shape'],scale = self.learning_params['parameter_prior_scale'])

    def Calculate_prior(self,particle:dict,flavour=''):
        H_opts = [a for a in list(particle.keys()) if a.startswith('H')]
        L_opts = [a for a in list(particle.keys()) if a.startswith('L')]
        complex_terms = len([s for s in L_opts if '+' in s])
        hamiltonian_terms = len(H_opts)
        lindbladian_terms = len(L_opts)
        model_prior = (1/(comb(self.all_lindblad_terms-self.all_complex_terms,lindbladian_terms-complex_terms)*comb(self.all_complex_terms,complex_terms)))*(1/comb(self.all_hamiltonian_terms,hamiltonian_terms))*np.exp(-hamiltonian_terms/self.learning_params['hamiltonian_exponential_rate'])*np.exp(-lindbladian_terms/self.learning_params['lindbladian_exponential_rate'])*np.exp(-1*complex_terms)
        parameter_prior = np.product([self.g.pdf(particle[k]) for k in particle.keys() if k != 'bkg'])
        if self.verbose:
            print(flavour + " Model Prior:\t\t\t" + str(model_prior) + '\n' + 
                flavour + " Parameter Prior:\t\t" + str(parameter_prior) + '\n')
        return model_prior, parameter_prior, (hamiltonian_terms,lindbladian_terms)

    @abstractmethod
    def Enact(self,operators,current_particle):
        '''
        In practice this function applies the move
        '''
        self.trial_particle = current_particle.copy()
        pass
    
    @abstractmethod
    def Accept_Reject(self,previous_particle,trial_probabiltiy, previous_probabilty,operators):
        '''
        The custom accept-reject rules for each move is handled in this move, 
        returning (posterior difference , new posterior )
        '''
        pass

class PARAMETER(Move):

    def Enact(self,operators,current_particle):
        '''
        This function re-samples a rate in the model from a gaussian centered on
        the rate's current value. The function is cautious of
        "immutable_processes" identified in config_file.py
        '''
        self.current_particle = current_particle
        self.trial_particle = self.current_particle.copy()
        op = choice(list(self.trial_particle.keys()))
        val = self.trial_particle[op]
        if op in self.learning_params['immutable_processes']:
            if self.learning_params['immutable_rates'][self.learning_params['immutable_processes'].index(op)] == None:
                if op == 'bkg':
                    self.trial_particle[op] = np.abs(np.random.normal(val,self.learning_params['initial_variance']/10))
                else:
                    self.trial_particle[op] = np.abs(np.random.normal(val,self.learning_params['initial_variance']))
            else:
                self.trial_particle = None                                
        else:
            self.trial_particle[op] = np.abs(np.random.normal(val,self.learning_params['initial_variance']))
        return self.trial_particle
    
    def Accept_Reject(self,previous_particle,trial_probabiltiy, previous_probabilty,operators):
        '''
        The accept-reject calculation is most simple for parameter update and
        only requires parameter priors as the structure of the model doesn't
        change. 

        This function can be used as a skeleton for new custome functions.
        '''
        if self.verbose:
            print('Previous particle: \t'+ ''.join([s for s in previous_particle.keys()]))
            print('Trial particle: \t'+ ''.join([s for s in self.trial_particle.keys()]))

        current_model_prior,current_parameter_prior,opts = self.Calculate_prior(previous_particle,flavour='Current ')
        trial_model_prior,trial_parameter_prior,_ = self.Calculate_prior(self.trial_particle,flavour='Trial \t')

        probability_of_chosen_move = 1
        probability_of_inverse_move = 1

        posterior_new = np.real(trial_probabiltiy+np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move))
        posterior_old = np.real(previous_probabilty+np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move))

        if self.verbose:
            print('\n' + 
                "probability_of_chosen_move: \t\t" + str(probability_of_chosen_move) + '\n' + 
                "probability_of_inverse_move: \t\t" + str(probability_of_inverse_move) + '\n' + 
                '\n' + 
                "Current log likelihood:: \t\t" + str(np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move)) + '\n' + 
                "Trial log likelihood: \t\t\t" + str(np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move)) + '\n' + 
                '\n' + 
                "current Experimental Meaurement: \t" + str(previous_probabilty) + '\n' + 
                "trial Experimental Meaurement: \t\t" + str(trial_probabiltiy) + '\n' + 
                '\n' + 
                "posterior_old: \t\t\t\t" + str(posterior_old) + '\n' + 
                "posterior_new: \t\t\t\t" + str(posterior_new))
        return (np.exp(posterior_new - posterior_old),posterior_new)
    
class ADDINGH(Move):

    def Enact(self, operators,current_particle):
        '''
        This function adds to a modela  single Hamiltonian process that is 
        selected by the new_operator utility function at the head of
        "learning_moves.py" and uses the gamma function defined in "config_file"
        to generate intial values.
        '''
        self.current_particle = current_particle
        self.trial_particle = self.current_particle.copy()
        op = new_operator(operators[0],'H',self.current_particle.keys())
        self.new_rate = np.random.gamma(self.learning_params['parameter_proposal_shape'],scale = self.learning_params['parameter_proposal_scale'])
        self.trial_particle[op] = self.new_rate
        return self.trial_particle
    
    def Accept_Reject(self, previous_particle, trial_probabiltiy, previous_probabilty,operators):
        '''
        Accept reject calculation here is more complicated as the model's
        dimension is changing however the model's prior is calculated in 
        "Calculate_prior" at the head of "learning_moves.py". 

        An exception is in place when attempting to make a model with ALL of the 
        possible hamiltonian operators. In this case the posterior is 
        artificially reduced to prevent models with all operators.
        '''
        if self.verbose:
            print('Previous particle: \t'+ ''.join([s for s in previous_particle.keys()]))
            print('Trial particle: \t'+ ''.join([s for s in self.trial_particle.keys()]))

        current_model_prior,current_parameter_prior,opts = self.Calculate_prior(previous_particle,flavour='Current ')
        trial_model_prior,trial_parameter_prior,_ = self.Calculate_prior(self.trial_particle,flavour='Trial \t')
        current_hamiltonian_terms = opts[0]

        probability_of_inverse_move = self.learning_params['REMOVINGH_chance']*(1/(current_hamiltonian_terms+1))
        try:
            probability_of_chosen_move = self.learning_params['ADDINGH_chance']*(1/(self.all_hamiltonian_terms-current_hamiltonian_terms))*self.g.pdf(self.new_rate)
        except:
            probability_of_chosen_move = 1e3
        posterior_new = np.real(trial_probabiltiy+np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move))
        posterior_old = np.real(previous_probabilty+np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move))

        if self.verbose:
            print('\n' + 
                "probability_of_chosen_move: \t\t" + str(probability_of_chosen_move) + '\n' + 
                "probability_of_inverse_move: \t\t" + str(probability_of_inverse_move) + '\n' + 
                '\n' + 
                "Current log likelihood:: \t\t" + str(np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move)) + '\n' + 
                "Trial log likelihood: \t\t\t" + str(np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move)) + '\n' + 
                '\n' + 
                "current Experimental Meaurement: \t" + str(previous_probabilty) + '\n' + 
                "trial Experimental Meaurement: \t\t" + str(trial_probabiltiy) + '\n' + 
                '\n' + 
                "posterior_old: \t\t\t\t" + str(posterior_old) + '\n' + 
                "posterior_new: \t\t\t\t" + str(posterior_new))
                
        return (np.exp(posterior_new - posterior_old),posterior_new)
    
class REMOVINGH(Move):

    def Enact(self, operators,current_particle):
        '''
        This function finds a random Hamiltonian process not in the 
        "immutable_processes" defined in "config_file.py" and returns the same 
        model with the process removed entirely.
        '''
        self.current_particle = current_particle
        if len([t for t in list(self.current_particle.keys()) if t.startswith('H') and t not in self.learning_params['immutable_processes']]) > 0:
            self.trial_particle = self.current_particle.copy()
            op = choice([t for t in list(self.trial_particle.keys()) if t.startswith('H') and t not in self.learning_params['immutable_processes']])
            self.new_rate = self.trial_particle[op]
            del self.trial_particle[op]
            return self.trial_particle
        else:
            return None
    
    def Accept_Reject(self, previous_particle, trial_probabiltiy, previous_probabilty,operators):
        '''
        Accept reject calculation here is more complicated as the model's
        dimension is changing however the model's prior is calculated in 
        "Calculate_prior" at the head of "learning_moves.py". 
        '''
        if self.verbose:
            print('Previous particle: \t'+ ''.join([s for s in previous_particle.keys()]))
            print('Trial particle: \t'+ ''.join([s for s in self.trial_particle.keys()]))

        current_model_prior,current_parameter_prior,opts = self.Calculate_prior(previous_particle,flavour='Current ')
        trial_model_prior,trial_parameter_prior,_ = self.Calculate_prior(self.trial_particle,flavour='Trial \t')
        current_hamiltonian_terms = opts[0]
        probability_of_inverse_move = self.learning_params['REMOVINGH_chance']*(1/(self.all_hamiltonian_terms-(current_hamiltonian_terms-1)))*self.g.pdf(self.new_rate)
        probability_of_chosen_move = self.learning_params['ADDINGH_chance']*(1/(current_hamiltonian_terms))

        posterior_new = np.real(trial_probabiltiy+np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move))
        posterior_old = np.real(previous_probabilty+np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move))
        if self.verbose:
            print('\n' + 
                "probability_of_chosen_move: \t\t" + str(probability_of_chosen_move) + '\n' + 
                "probability_of_inverse_move: \t\t" + str(probability_of_inverse_move) + '\n' + 
                '\n' + 
                "Current log likelihood:: \t\t" + str(np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move)) + '\n' + 
                "Trial log likelihood: \t\t\t" + str(np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move)) + '\n' + 
                '\n' + 
                "current Experimental Meaurement: \t" + str(previous_probabilty) + '\n' + 
                "trial Experimental Meaurement: \t\t" + str(trial_probabiltiy) + '\n' + 
                '\n' + 
                "posterior_old: \t\t\t\t" + str(posterior_old) + '\n' + 
                "posterior_new: \t\t\t\t" + str(posterior_new))
            
        return (np.exp(posterior_new - posterior_old),posterior_new)
    
class SWAPPINGH(Move):

    def Enact(self,operators,current_particle):
        '''
        A model is taken in and a Hamiltonian term is selected. This process is
        removed and replaced with another Hamiltonian term. The rate for this
        process is re-sampled from a gamma distribution defined in 
        "config_file.py". This process is mindful of "immutable_processes"
        defined in "config_file.py"
        '''
        self.current_particle = current_particle
        if len([t for t in list(self.current_particle.keys()) if t.startswith('H') and t not in self.learning_params['immutable_processes']]) > 0:
            self.trial_particle = self.current_particle.copy()
            op_new = new_operator(operators[0],'H',self.current_particle.keys())
            op_old = choice([t for t in list(self.trial_particle.keys()) if t.startswith('H') and t not in self.learning_params['immutable_processes']])
            del self.trial_particle[op_old]
            self.trial_particle[op_new] = np.random.gamma(self.learning_params['parameter_proposal_shape'],scale = self.learning_params['parameter_proposal_scale'])
            return self.trial_particle
        else:
            return None
    
    def Accept_Reject(self,previous_particle,trial_probabiltiy, previous_probabilty,operators):
        '''
        A swapping operation does not change the structure of the model and so
        this too is a simplified accept-reject calculation.
        '''
        if self.verbose:
            print('Previous particle: \t'+ ''.join([s for s in previous_particle.keys()]))
            print('Trial particle: \t'+ ''.join([s for s in self.trial_particle.keys()]))

        current_model_prior,current_parameter_prior,opts = self.Calculate_prior(previous_particle,flavour='Current ')
        trial_model_prior,trial_parameter_prior,_ = self.Calculate_prior(self.trial_particle,flavour='Trial \t')

        probability_of_chosen_move = 1
        probability_of_inverse_move = 1

        posterior_new = np.real(trial_probabiltiy+np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move))
        posterior_old = np.real(previous_probabilty+np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move))

        if self.verbose:
            print('\n' + 
                "probability_of_chosen_move: \t\t" + str(probability_of_chosen_move) + '\n' + 
                "probability_of_inverse_move: \t\t" + str(probability_of_inverse_move) + '\n' + 
                '\n' + 
                "Current log likelihood:: \t\t" + str(np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move)) + '\n' + 
                "Trial log likelihood: \t\t\t" + str(np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move)) + '\n' + 
                '\n' + 
                "current Experimental Meaurement: \t" + str(trial_probabiltiy) + '\n' + 
                "trial Experimental Meaurement: \t\t" + str(trial_probabiltiy) + '\n' + 
                '\n' + 
                "posterior_old: \t\t\t\t" + str(posterior_old) + '\n' + 
                "posterior_new: \t\t\t\t" + str(posterior_new))
        return (np.exp(posterior_new - posterior_old),posterior_new)
    
class ADDINGL_SINGLE(Move):

    def Enact(self, operators,current_particle):
        '''
        This function adds to a modela  single Lindbladian process that is 
        selected by the new_operator utility function at the head of
        "learning_moves.py" and uses the gamma function defined in "config_file"
        to generate intial values.
        '''
        self.current_particle = current_particle
        self.trial_particle = self.current_particle.copy()
        op = new_operator(operators[1],'L',self.current_particle.keys())
        print('Chosen op:',op)
        self.new_rate = np.random.gamma(self.learning_params['parameter_proposal_shape'],scale = self.learning_params['parameter_proposal_scale'])
        self.trial_particle[op] = self.new_rate
        return self.trial_particle
    
    def Accept_Reject(self, previous_particle, trial_probabiltiy, previous_probabilty,operators):
        '''
        Accept reject calculation here is more complicated as the model's
        dimension is changing however the model's prior is calculated in 
        "Calculate_prior" at the head of "learning_moves.py". 

        An exception is in place when attempting to make a model with ALL of the 
        possible Lindbladian operators. In this case the posterior is 
        artificially reduced to prevent models with all operators.
        '''
        if self.verbose:
            print('Previous particle: \t'+ ''.join([s for s in previous_particle.keys()]))
            print('Trial particle: \t'+ ''.join([s for s in self.trial_particle.keys()]))

        current_model_prior,current_parameter_prior,opts = self.Calculate_prior(previous_particle,flavour='Current ')
        trial_model_prior,trial_parameter_prior,_ = self.Calculate_prior(self.trial_particle,flavour='Trial \t')
        current_lindbladian_terms = opts[1]

        probability_of_inverse_move = self.learning_params['REMOVINGL_SINGLE_chance']*(1/(current_lindbladian_terms+1))
        probability_of_chosen_move = self.learning_params['ADDINGL_SINGLE_chance']*(1/(self.all_lindblad_terms-current_lindbladian_terms))*self.g.pdf(self.new_rate)
        
        posterior_new = np.real(trial_probabiltiy+np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move))
        posterior_old = np.real(previous_probabilty+np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move))

        if self.verbose:
            print('\n' + 
                "probability_of_chosen_move: \t\t" + str(probability_of_chosen_move) + '\n' + 
                "probability_of_inverse_move: \t\t" + str(probability_of_inverse_move) + '\n' + 
                '\n' + 
                "Current log likelihood:: \t\t" + str(np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move)) + '\n' + 
                "Trial log likelihood: \t\t\t" + str(np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move)) + '\n' + 
                '\n' + 
                "current Experimental Meaurement: \t" + str(previous_probabilty) + '\n' + 
                "trial Experimental Meaurement: \t\t" + str(trial_probabiltiy) + '\n' + 
                '\n' + 
                "posterior_old: \t\t\t\t" + str(posterior_old) + '\n' + 
                "posterior_new: \t\t\t\t" + str(posterior_new))
                
        return (np.exp(posterior_new - posterior_old),posterior_new)

class REMOVINGL_SINGLE(Move):

    def Enact(self, operators,current_particle):
        '''
        This function finds a random Lindbladian process not in the 
        "immutable_processes" defined in "config_file.py" and returns the same 
        model with the process removed entirely.
        '''
        self.current_particle = current_particle
    
        if len([t for t in list(self.current_particle.keys()) if t.startswith('L') and t not in self.learning_params['immutable_processes']]) > 0:
            self.trial_particle = self.current_particle.copy()
            op = choice([t for t in list(self.trial_particle.keys()) if t.startswith('L') and t not in self.learning_params['immutable_processes']])
            self.new_rate = self.trial_particle[op]
            del self.trial_particle[op]
            return self.trial_particle
        else:
            return None
    
    def Accept_Reject(self, previous_particle, trial_probabiltiy, previous_probabilty,operators):
        '''
        Accept reject calculation here is more complicated as the model's
        dimension is changing however the model's prior is calculated in 
        "Calculate_prior" at the head of "learning_moves.py". 
        '''
        if self.verbose:
            print('Previous particle: \t'+ ''.join([s for s in previous_particle.keys()]))
            print('Trial particle: \t'+ ''.join([s for s in self.trial_particle.keys()]))

        current_model_prior,current_parameter_prior,opts = self.Calculate_prior(previous_particle,flavour='Current ')
        trial_model_prior,trial_parameter_prior,_ = self.Calculate_prior(self.trial_particle,flavour='Trial \t')
        current_lindbladian_terms = opts[1]

        probability_of_inverse_move = self.learning_params['ADDINGL_SINGLE_chance']*(1/(self.all_lindblad_terms-(current_lindbladian_terms-1)))*self.g.pdf(self.new_rate)
        probability_of_chosen_move = self.learning_params['REMOVINGL_SINGLE_chance']*(1/(current_lindbladian_terms))

        posterior_new = np.real(trial_probabiltiy+np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move))
        posterior_old = np.real(previous_probabilty+np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move))
        if self.verbose:
            print('\n' + 
                "probability_of_chosen_move: \t\t" + str(probability_of_chosen_move) + '\n' + 
                "probability_of_inverse_move: \t\t" + str(probability_of_inverse_move) + '\n' + 
                '\n' + 
                "Current log likelihood:: \t\t" + str(np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move)) + '\n' + 
                "Trial log likelihood: \t\t\t" + str(np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move)) + '\n' + 
                '\n' + 
                "current Experimental Meaurement: \t" + str(previous_probabilty) + '\n' + 
                "trial Experimental Meaurement: \t\t" + str(trial_probabiltiy) + '\n' + 
                '\n' + 
                "posterior_old: \t\t\t\t" + str(posterior_old) + '\n' + 
                "posterior_new: \t\t\t\t" + str(posterior_new))
            
        return (np.exp(posterior_new - posterior_old),posterior_new)
    
class SWAPPINGL_SINGLE(Move):

    def Enact(self,operators,current_particle):
        '''
        A model is taken in and a Lindbladian term is selected. This process is
        removed and replaced with another Lindbladian term. The rate for this
        process is re-sampled from a gamma distribution defined in 
        "config_file.py". This process is mindful of "immutable_processes"
        defined in "config_file.py"
        '''
        self.current_particle = current_particle
        if len([t for t in list(self.current_particle.keys()) if t.startswith('L') and t not in self.learning_params['immutable_processes']]) > 0:
            self.trial_particle = self.current_particle.copy()
            op_new = new_operator(operators[1],'L',self.current_particle.keys())
            op_old = choice([t for t in list(self.trial_particle.keys()) if t.startswith('L') and t not in self.learning_params['immutable_processes']])
            del self.trial_particle[op_old]
            self.trial_particle[op_new] = np.random.gamma(self.learning_params['parameter_proposal_shape'],scale = self.learning_params['parameter_proposal_scale'])
            return self.trial_particle
        else:
            return None
    
    def Accept_Reject(self,previous_particle,trial_probabiltiy, previous_probabilty,operators):
        '''
        A swapping operation does not change the structure of the model and so
        this too is a simplified accept-reject calculation.
        '''
        if self.verbose:
            print('Previous particle: \t'+ ''.join([s for s in previous_particle.keys()]))
            print('Trial particle: \t'+ ''.join([s for s in self.trial_particle.keys()]))

        current_model_prior,current_parameter_prior,opts = self.Calculate_prior(previous_particle,flavour='Current ')
        trial_model_prior,trial_parameter_prior,_ = self.Calculate_prior(self.trial_particle,flavour='Trial \t')

        probability_of_chosen_move = 1
        probability_of_inverse_move = 1

        posterior_new = np.real(trial_probabiltiy+np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move))
        posterior_old = np.real(previous_probabilty+np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move))

        if self.verbose:
            print('\n' + 
                "probability_of_chosen_move: \t\t" + str(probability_of_chosen_move) + '\n' + 
                "probability_of_inverse_move: \t\t" + str(probability_of_inverse_move) + '\n' + 
                '\n' + 
                "Current log likelihood:: \t\t" + str(np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move)) + '\n' + 
                "Trial log likelihood: \t\t\t" + str(np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move)) + '\n' + 
                '\n' + 
                "current Experimental Meaurement: \t" + str(previous_probabilty) + '\n' + 
                "trial Experimental Meaurement: \t\t" + str(trial_probabiltiy) + '\n' + 
                '\n' + 
                "posterior_old: \t\t\t\t" + str(posterior_old) + '\n' + 
                "posterior_new: \t\t\t\t" + str(posterior_new))
        return (np.exp(posterior_new - posterior_old),posterior_new)
    
class ADDINGL_PAIR(Move):

    def Enact(self, operators,current_particle):
        '''
        In this function a sets of processes which fulfil the condition that 
        a* = b where * denotes transpose conjugate and either a or b are in the 
        model. This check is preformed by the function "conjugate_string" defined
        at the top of "learning_moves.py".

        If a is present in the model this function returns the same model with
        process b added. The rates are sampled from the default gamma
        distribution defined in "config_file.py"
        '''
        self.current_particle = current_particle

        self.paired_conj_terms = 0
        self.unmatched_paired_terms = 0
        unmatched_pairs = []

        for i in [t for t in list(self.current_particle.keys()) if t.startswith('L') and t not in self.learning_params['immutable_processes']]:
            if 'a' in i or 's' in i:
                if conjugate_string(i,operators[1]) in self.current_particle.keys():
                    self.paired_conj_terms += 1
                else:
                    self.unmatched_paired_terms += 1
                    unmatched_pairs.append(i)

        if self.unmatched_paired_terms > 0:
            self.trial_particle = self.current_particle.copy()
            op = conjugate_string(choice(unmatched_pairs),operators[1])
            self.new_rate = np.random.gamma(self.learning_params['parameter_proposal_shape'],scale = self.learning_params['parameter_proposal_scale'])
            self.trial_particle[op] = self.new_rate
            return self.trial_particle
        else:
            return None
        
    def Accept_Reject(self, previous_particle, trial_probabiltiy, previous_probabilty,operators):
        '''
        This function is more complex than the others because it is informed 
        by a subset of processes and these values change during learning. The
        publication (TODO link paper) covers this calculation in more detail.
        '''
        if self.verbose:
            print('Previous particle: \t'+ ''.join([s for s in previous_particle.keys()]))
            print('Trial particle: \t'+ ''.join([s for s in self.trial_particle.keys()]))

        current_model_prior,current_parameter_prior,opts = self.Calculate_prior(previous_particle,flavour='Current ')
        trial_model_prior,trial_parameter_prior,_ = self.Calculate_prior(self.trial_particle,flavour='Trial \t')

        probability_of_inverse_move = self.learning_params['REMOVINGL_PAIR_chance']*(1/(self.paired_conj_terms+2))
        probability_of_chosen_move = self.learning_params['ADDINGL_PAIR_chance']*(1/self.unmatched_paired_terms)*self.g.pdf(self.new_rate)

        posterior_new = np.real(trial_probabiltiy+np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move))
        posterior_old = np.real(previous_probabilty+np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move))
        if self.verbose:
            print('\n' + 
                "probability_of_chosen_move: \t\t" + str(probability_of_chosen_move) + '\n' + 
                "probability_of_inverse_move: \t\t" + str(probability_of_inverse_move) + '\n' + 
                '\n' + 
                "Current log likelihood:: \t\t" + str(np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move)) + '\n' + 
                "Trial log likelihood: \t\t\t" + str(np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move)) + '\n' + 
                '\n' + 
                "current Experimental Meaurement: \t" + str(previous_probabilty) + '\n' + 
                "trial Experimental Meaurement: \t\t" + str(trial_probabiltiy) + '\n' + 
                '\n' + 
                "posterior_old: \t\t\t\t" + str(posterior_old) + '\n' + 
                "posterior_new: \t\t\t\t" + str(posterior_new))
        return (np.exp(posterior_new - posterior_old),posterior_new)
                
class REMOVINGL_PAIR(Move):

    def Enact(self, operators,current_particle):
        '''
        In this function a sets of processes which fulfil the condition that 
        a* = b where * denotes transpose conjugate and both a and b are in the 
        model. This check is preformed by the function "conjugate_string" defined
        at the top of "learning_moves.py".

        If a and b are present in the model then this fucntion randomly selects 
        one of the pairs of a and b's and fully removes one of them.
        '''
        self.current_particle = current_particle

        self.paired_conj_terms = 0
        self.unmatched_paired_terms = 0
        matched_pairs = []

        for i in [t for t in list(self.current_particle.keys()) if t.startswith('L') and t not in self.learning_params['immutable_processes']]:
            if 'a' in i or 's' in i:
                if conjugate_string(i,operators[1]) in self.current_particle.keys() and conjugate_string(i,operators[1]) != i:
                    self.paired_conj_terms += 1
                    matched_pairs.append(i)
                else:
                    self.unmatched_paired_terms += 1
                if conjugate_string(i,operators[1]) in self.current_particle.keys() and conjugate_string(i,operators[1]) == i and self.verbose:
                    print('Hermitian Operator Found ',i)
                    

        if self.paired_conj_terms > 0:
            self.trial_particle = self.current_particle.copy()
            op = choice(matched_pairs)

            self.new_rate = self.trial_particle[op]
            del self.trial_particle[op]
            return self.trial_particle
        else:
            return None
        
    def Accept_Reject(self, previous_particle, trial_probabiltiy, previous_probabilty,operators):
        '''
        This function is more complex than the others because it is informed 
        by a subset of processes and these values change during learning. The
        publication (TODO link paper) covers this calculation in more detail.
        '''
        if self.verbose:
            print('Previous particle: \t'+ ''.join([s for s in previous_particle.keys()]))
            print('Trial particle: \t'+ ''.join([s for s in self.trial_particle.keys()]))

        current_model_prior,current_parameter_prior,opts = self.Calculate_prior(previous_particle,flavour='Current ')
        trial_model_prior,trial_parameter_prior,_ = self.Calculate_prior(self.trial_particle,flavour='Trial \t')
        current_lindbladian_terms = opts[1]

        probability_of_inverse_move = self.learning_params['ADDINGL_PAIR_chance']*(1/(self.unmatched_paired_terms+1))*self.g.pdf(self.new_rate)
        probability_of_chosen_move = self.learning_params['REMOVINGL_PAIR_chance']*(1/(self.paired_conj_terms))

        posterior_new = np.real(trial_probabiltiy+np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move))
        posterior_old = np.real(previous_probabilty+np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move))

        if self.verbose:
            print('\n' + 
                "probability_of_chosen_move: \t\t" + str(probability_of_chosen_move) + '\n' + 
                "probability_of_inverse_move: \t\t" + str(probability_of_inverse_move) + '\n' + 
                '\n' + 
                "Current log likelihood:: \t\t" + str(np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move)) + '\n' + 
                "Trial log likelihood: \t\t\t" + str(np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move)) + '\n' + 
                '\n' + 
                "current Experimental Meaurement: \t" + str(previous_probabilty) + '\n' + 
                "trial Experimental Meaurement: \t\t" + str(trial_probabiltiy) + '\n' + 
                '\n' + 
                "posterior_old: \t\t\t\t" + str(posterior_old) + '\n' + 
                "posterior_new: \t\t\t\t" + str(posterior_new))
        return (np.exp(posterior_new - posterior_old),posterior_new)
    
class SWAPPINGL_PAIR(Move):

    def Enact(self,operators,current_particle):
        '''
        In this function a sets of processes which fulfil the condition that 
        a* = b where * denotes transpose conjugate and both a and b are in the 
        model. This check is preformed by the function "conjugate_string" defined
        at the top of "learning_moves.py".

        This function finds all of the pairs of a and b that fit the formula 
        above and then randomly selects on of the pairs. The pair is fully 
        removed and another random pair of a and b processes, neighter of which 
        are currently present in the model ar ehten added to the model with new 
        rates which are selected from a gamma function. This function is defined 
        in the "config_file.py" file.
        '''
        if self.verbose:
            print(current_particle)
        self.current_particle = current_particle

        self.paired_conj_terms = 0
        self.unmatched_paired_terms = 0
        matched_pairs = []

        for i in [t for t in list(self.current_particle.keys()) if t.startswith('L') and t not in self.learning_params['immutable_processes']]:
            if 'a' in i or 's' in i:
                if conjugate_string(i,operators[1]) in self.current_particle.keys():
                    self.paired_conj_terms += 1
                    matched_pairs.append(i)
                else:
                    self.unmatched_paired_terms += 1
        if self.verbose:
            print(current_particle)
        if len(matched_pairs) > 0:
            self.trial_particle = self.current_particle.copy()
            op_old_0 = choice(matched_pairs)
            op_old_1 = conjugate_string(op_old_0,operators[1])
            if op_old_0 == op_old_1:
                return None

            non_hermitian_terms = ['L'+o for o in list(operators[1].keys()) if 'a' not in o and 's' not in o]
            op_new_0 = new_operator(operators[1],flavour='L', exc = [t for t in list(self.current_particle.keys()) if t.startswith('L')] + non_hermitian_terms)
            op_new_1 = conjugate_string(op_new_0,operators[1])



            del self.trial_particle[op_old_0]
            del self.trial_particle[op_old_1]

            self.trial_particle[op_new_0] = np.random.gamma(self.learning_params['parameter_proposal_shape'],scale = self.learning_params['parameter_proposal_scale'])
            self.trial_particle[op_new_1] = np.random.gamma(self.learning_params['parameter_proposal_shape'],scale = self.learning_params['parameter_proposal_scale'])
            return self.trial_particle
        else:
            return None
    
    def Accept_Reject(self,previous_particle,trial_probabiltiy, previous_probabilty,operators):
        '''
        This function is more complex than the others because it is informed 
        by a subset of processes and these values change during learning. The
        publication (TODO link paper) covers this calculation in more detail.
        '''
        if self.verbose:
            print('Previous particle: \t'+ ''.join([s for s in previous_particle.keys()]))
            print('Trial particle: \t'+ ''.join([s for s in self.trial_particle.keys()]))

        current_model_prior,current_parameter_prior,opts = self.Calculate_prior(previous_particle,flavour='Current ')
        trial_model_prior,trial_parameter_prior,_ = self.Calculate_prior(self.trial_particle,flavour='Trial \t')

        probability_of_chosen_move = 1
        probability_of_inverse_move = 1

        posterior_new = np.real(trial_probabiltiy+np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move))
        posterior_old = np.real(previous_probabilty+np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move))

        if self.verbose:
            print('\n' + 
                "probability_of_chosen_move: \t\t" + str(probability_of_chosen_move) + '\n' + 
                "probability_of_inverse_move: \t\t" + str(probability_of_inverse_move) + '\n' + 
                '\n' + 
                "Current log likelihood:: \t\t" + str(np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move)) + '\n' + 
                "Trial log likelihood: \t\t\t" + str(np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move)) + '\n' + 
                '\n' + 
                "current Experimental Meaurement: \t" + str(previous_probabilty) + '\n' + 
                "trial Experimental Meaurement: \t\t" + str(trial_probabiltiy) + '\n' + 
                '\n' + 
                "posterior_old: \t\t\t\t" + str(posterior_old) + '\n' + 
                "posterior_new: \t\t\t\t" + str(posterior_new))
        return (np.exp(posterior_new - posterior_old),posterior_new)
    
class BASIS(Move):
    def __init__(self,learning_params,operators,verbose = False):
        '''
        The basis move uses outside data run at the intialisation of the 
        learning to assess what unitary transformations are allowed in the 
        learning. This is all parsed in the intialisation of the function.
        '''
        super().__init__(learning_params,operators,verbose)
        directory = getcwd()
        print(directory)
        with open(learning_params['operator path'][:-4]+'_unitary_set.npy') as f:
            self.unitary_options = load(f)
            f.close()
        all_keys = set([k for u in self.unitary_options for k in u.keys()])
        all_values = set([k for u in self.unitary_options for k in u.values()])
        for k in all_keys:
            if k in all_values:
                pass
            else:
                print(k, ' missing')


    def Enact(self, operators, current_particle):
        '''
        The function randomly selects one of the "self.unitary_options" which 
        are all of the unitary transformations identified byt the 
        "uniform_search" function in "uniform_search.py" which is run at the 
        intialisation of the learning agent. For a transofmration to be present 
        in "self.unitary_dict" all possible operators in the models must be able 
        to be cast to other processes. 

        This function takes a model and applies one of the unitary transforms to 
        each operator, leaving the rates the same.
        '''
        translation_dict = choice(self.unitary_options)
        original_keys = list(current_particle.keys())
        self.trial_particle = {}
        for k in original_keys:
            if k in self.learning_params['immutable_processes']:
                self.trial_particle[k] = current_particle[k]
            else:
                self.trial_particle[k[0]+translation_dict[k[1:]]] = current_particle[k]
        return self.trial_particle
    
    def Accept_Reject(self, previous_particle, trial_probabiltiy, previous_probabilty, operators):
        '''
        The accept reject for a basis swap is straight forward as the number of 
        terms cannot change during this step.
        '''
        if self.verbose:
            print('Previous particle: \t'+ ''.join([s for s in previous_particle.keys()]))
            print('Trial particle: \t'+ ''.join([s for s in self.trial_particle.keys()]))

        current_model_prior,current_parameter_prior,opts = self.Calculate_prior(previous_particle,flavour='Current ')
        trial_model_prior,trial_parameter_prior,_ = self.Calculate_prior(self.trial_particle,flavour='Trial \t')

        probability_of_chosen_move = 1
        probability_of_inverse_move = 1

        posterior_new = np.real(trial_probabiltiy+np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move))
        posterior_old = np.real(previous_probabilty+np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move))

        if self.verbose:
            print('\n' + 
                "probability_of_chosen_move: \t\t" + str(probability_of_chosen_move) + '\n' + 
                "probability_of_inverse_move: \t\t" + str(probability_of_inverse_move) + '\n' + 
                '\n' + 
                "Current log likelihood:: \t\t" + str(np.log(current_model_prior*current_parameter_prior*probability_of_chosen_move)) + '\n' + 
                "Trial log likelihood: \t\t\t" + str(np.log(trial_model_prior*trial_parameter_prior*probability_of_inverse_move)) + '\n' + 
                '\n' + 
                "current Experimental Meaurement: \t" + str(previous_probabilty) + '\n' + 
                "trial Experimental Meaurement: \t\t" + str(trial_probabiltiy) + '\n' + 
                '\n' + 
                "posterior_old: \t\t\t\t" + str(posterior_old) + '\n' + 
                "posterior_new: \t\t\t\t" + str(posterior_new))
        return (np.exp(posterior_new - posterior_old),posterior_new)
    