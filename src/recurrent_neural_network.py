'''
Created on Jun 3, 2013

@author: sumanravuri
'''

import sys
import numpy as np
import scipy.io as sp
import scipy.linalg as sl
import scipy.optimize as sopt
import math
import copy
import argparse
from deep_neural_network import Vector_Math

class Recurrent_Neural_Network_Weight(object):
    def __init__(self, init_hiddens = None, weights=None, bias=None, weight_type=None):
        """num_layers
        weights - actual Recurrent Neural Network weights, a dictionary with keys corresponding to layer, ie. weights['visible_hidden'], weights['hidden_hidden'], and weights['hidden_output'] each numpy array
        bias - NN biases, again a dictionary stored as bias['visible'], bias['hidden'], bias['output'], etc.
        weight_type - optional command indexed by same keys weights, possible optionals are 'rbm_gaussian_bernoullli', 'rbm_bernoulli_bernoulli'"""
        self.valid_layer_types = dict()
        self.valid_layer_types['visible_hidden'] = ['rbm_gaussian_bernoulli', 'rbm_bernoulli_bernoulli']
        self.valid_layer_types['hidden_hidden'] = ['rbm_bernoulli_bernoulli']
        self.valid_layer_types['hidden_output'] = ['logistic']
        self.bias_keys = ['visible', 'hidden', 'output']
        self.weights_keys = ['visible_hidden', 'hidden_hidden', 'hidden_output']
        
        if init_hiddens != None:
            self.init_hiddens = init_hiddens
        if weights == None:
            self.weights = dict()
        else:
            self.weights = copy.deepcopy(weights)
        if bias == None:
            self.bias = dict()
        else:
            self.bias = copy.deepcopy(bias)
        if weight_type == None:
            self.weight_type = dict()
        else:
            self.weight_type = copy.deepcopy(weight_type)
    def clear(self):
        self.num_layers = 0
        self.weights.clear()
        self.bias.clear()
        self.weight_type.clear()
    def dot(self, nn_weight2, excluded_keys = {'bias': [], 'weights': []}):
        if type(nn_weight2) is not Recurrent_Neural_Network_Weight:
            print "argument must be of type Recurrent_Neural_Network_Weight... instead of type", type(nn_weight2), "Exiting now..."
            sys.exit()
        return_val = 0
        for key in self.bias_keys:
            if key in excluded_keys['bias']:
                continue
            return_val += np.sum(self.bias[key] * nn_weight2.bias[key])
        for key in self.weights_keys:
            if key in excluded_keys['weights']:
                continue
            return_val += np.sum(self.weights[key] * nn_weight2.weights[key])
        return return_val
    def __str__(self):
        string = ""
        for key in self.bias_keys:
            string = string + "bias key " + key + "\n"
            string = string + str(self.bias[key]) + "\n"
        for key in self.weights_keys:
            string = string + "weight key " + key + "\n"
            string = string + str(self.weights[key]) + "\n"
        return string
    def print_statistics(self):
        for key in self.bias_keys:
            print "min of bias[" + key + "] is", np.min(self.bias[key]) 
            print "max of bias[" + key + "] is", np.max(self.bias[key])
            print "mean of bias[" + key + "] is", np.mean(self.bias[key])
            print "var of bias[" + key + "] is", np.var(self.bias[key]), "\n"
        for key in self.weights_keys:
            print "min of weights[" + key + "] is", np.min(self.weights[key]) 
            print "max of weights[" + key + "] is", np.max(self.weights[key])
            print "mean of weights[" + key + "] is", np.mean(self.weights[key])
            print "var of weights[" + key + "] is", np.var(self.weights[key]), "\n"
        
        print "min of init_hiddens is", np.min(self.init_hiddens) 
        print "max of init_hiddens is", np.max(self.init_hiddens)
        print "mean of init_hiddens is", np.mean(self.init_hiddens)
        print "var of init_hiddens is", np.var(self.init_hiddens), "\n"
        
    def norm(self, excluded_keys = {'bias': [], 'weights': []}, calc_init_hiddens=False):
        squared_sum = 0
        for key in self.bias_keys:
            if key in excluded_keys['bias']:
                continue
            squared_sum += np.sum(self.bias[key] ** 2)
        for key in self.weights_keys:
            if key in excluded_keys['weights']:
                continue  
            squared_sum += np.sum(self.weights[key] ** 2)
        if calc_init_hiddens:
            squared_sum += np.sum(self.init_hiddens ** 2)
        return np.sqrt(squared_sum)
    def max(self, excluded_keys = {'bias': [], 'weights': []}, calc_init_hiddens=False):
        max_val = -float('Inf')
        for key in self.bias_keys:
            if key in excluded_keys['bias']:
                continue
            max_val = max(np.max(self.bias[key]), max_val)
        for key in self.weights_keys:
            if key in excluded_keys['weights']:
                continue  
            max_val = max(np.max(self.weights[key]), max_val)
        if calc_init_hiddens:
            max_val = max(np.max(self.init_hiddens), max_val)
        return max_val
    def min(self, excluded_keys = {'bias': [], 'weights': []}, calc_init_hiddens=False):
        min_val = float('Inf')
        for key in self.bias_keys:
            if key in excluded_keys['bias']:
                continue
            min_val = min(np.min(self.bias[key]), min_val)
        for key in self.weights_keys:
            if key in excluded_keys['weights']:
                continue  
            min_val = min(np.min(self.weights[key]), min_val)
        if calc_init_hiddens:
            min_val = min(np.min(self.init_hiddens), min_val)
        return min_val
    def clip(self, clip_min, clip_max, excluded_keys = {'bias': [], 'weights': []}, calc_init_hiddens=False):
        nn_output = copy.deepcopy(self)
        for key in self.bias_keys:
            if key in excluded_keys['bias']:
                continue
            np.clip(self.bias[key], clip_min, clip_max, out=nn_output.bias[key])
        for key in self.weights_keys:
            if key in excluded_keys['weights']:
                continue  
            np.clip(self.weights[key], clip_min, clip_max, out=nn_output.weights[key])
        if calc_init_hiddens:
            np.clip(self.init_hiddens, clip_min, clip_max, out=nn_output.init_hiddens)
        return nn_output
    def get_architecture(self):
        return [self.bias['visible'].size, self.bias['hidden'].size, self.bias['output'].size]
    def size(self, excluded_keys = {'bias': [], 'weights': []}, include_init_hiddens=True):
        numel = 0
        for key in self.bias_keys:
            if key in excluded_keys['bias']:
                continue
            numel += self.bias[key].size
        for key in self.weights_keys:
            if key in excluded_keys['weights']:
                continue  
            numel += self.weights[key].size
        if include_init_hiddens:
            numel += self.init_hiddens.size
        return numel
    def open_weights(self, weight_matrix_name): #completed
        """the weight file format is very specific, it contains the following variables:
        weights_visible_hidden, weights_hidden_hidden, weights_hidden_output,
        bias_visible, bias_hidden, bias_output,
        init_hiddens, 
        weights_visible_hidden_type, weights_hidden_hidden_type, weights_hidden_output_type, etc...
        everything else will be ignored"""
        try:
            weight_dict = sp.loadmat(weight_matrix_name)
        except IOError:
            print "Unable to open", weight_matrix_name, "exiting now"
            sys.exit()
        try:
            self.bias['visible'] = weight_dict['bias_visible']
        except KeyError:
            print "bias_visible not found. bias_visible must exist for", weight_matrix_name, "to be a valid weight file... Exiting now"
            sys.exit()
        
        try:
            self.bias['hidden'] = weight_dict['bias_hidden']
        except KeyError:
            print "bias_hidden not found. bias_hidden must exist for", weight_matrix_name, "to be a valid weight file... Exiting now"
            sys.exit()
        
        try:
            self.bias['output'] = weight_dict['bias_output']
        except KeyError:
            print "bias_output not found. bias_output must exist for", weight_matrix_name, "to be a valid weight file... Exiting now"
            sys.exit()
        
        #TODO: dump these inside of try
        self.init_hiddens = weight_dict['init_hiddens']
        self.weights['visible_hidden'] = weight_dict['weights_visible_hidden']
        self.weights['hidden_hidden'] = weight_dict['weights_hidden_hidden']
        self.weights['hidden_output'] = weight_dict['weights_hidden_output']
        
        self.weight_type['visible_hidden'] = weight_dict['weights_visible_hidden_type'].encode('ascii', 'ignore')
        self.weight_type['hidden_hidden'] = weight_dict['weights_hidden_hidden_type'].encode('ascii', 'ignore')
        self.weight_type['hidden_output'] = weight_dict['weights_hidden_output_type'].encode('ascii', 'ignore')
        
        del weight_dict
        self.check_weights()
    def init_random_weights(self, architecture, initial_bias_max, initial_bias_min, initial_weight_max, initial_weight_min, seed=0): #completed, expensive, should be compiled
        np.random.seed(seed)
        
        initial_bias_range = initial_bias_max - initial_bias_min
        initial_weight_range = initial_weight_max - initial_weight_min
        self.bias['visible'] = initial_bias_min + initial_bias_range * np.random.random_sample((1,architecture[0]))
        self.bias['hidden'] = initial_bias_min + initial_bias_range * np.random.random_sample((1,architecture[1]))
        self.bias['output'] = initial_bias_min + initial_bias_range * np.random.random_sample((1,architecture[2]))
        
        self.init_hiddens = -1.0 + 2.0 * np.random.random_sample((1,architecture[1])) #because of tanh non-linearity
        
        self.weights['visible_hidden']=(initial_weight_min + initial_weight_range * 
                                        np.random.random_sample( (architecture[0],architecture[1]) ))
        self.weights['hidden_hidden']=(initial_weight_min + initial_weight_range * 
                                       np.random.random_sample( (architecture[1],architecture[1]) ))
        self.weights['hidden_output']=(initial_weight_min + initial_weight_range * 
                                       np.random.random_sample( (architecture[1],architecture[2]) ))
        
        self.weight_type['visible_hidden'] = 'rbm_gaussian_bernoulli'
        self.weight_type['hidden_hidden'] = 'rbm_bernoulli_bernoulli'
        self.weight_type['hidden_output'] = 'logistic'
        
        print "Finished Initializing Weights"
        self.check_weights()
    def init_zero_weights(self, architecture, verbose=False):
        self.init_hiddens = np.zeros((1,architecture[1]))
        self.bias['visible'] = np.zeros((1,architecture[0]))
        self.bias['hidden'] = np.zeros((1,architecture[1]))
        self.bias['output'] = np.zeros((1,architecture[2]))
        
        self.init_hiddens = np.zeros((1,architecture[1]))
         
        self.weights['visible_hidden'] = np.zeros( (architecture[0],architecture[1]) )
        self.weights['hidden_hidden'] = np.zeros( (architecture[1],architecture[1]) )
        self.weights['hidden_output'] = np.zeros( (architecture[1],architecture[2]) )
        
        self.weight_type['visible_hidden'] = 'rbm_gaussian_bernoulli'
        self.weight_type['hidden_hidden'] = 'rbm_bernoulli_bernoulli'
        self.weight_type['hidden_output'] = 'logistic'
        if verbose:
            print "Finished Initializing Weights"
        self.check_weights(False)
    def check_weights(self, verbose=True): #need to check consistency of features with weights
        """checks weights to see if following conditions are true
        *feature dimension equal to number of rows of first layer (if weights are stored in n_rows x n_cols)
        *n_cols of (n-1)th layer == n_rows of nth layer
        if only one layer, that weight layer type is logistic, gaussian_bernoulli or bernoulli_bernoulli
        check is biases match weight values
        if multiple layers, 0 to (n-1)th layer is gaussian bernoulli RBM or bernoulli bernoulli RBM and last layer is logistic regression
        """
        if verbose:
            print "Checking weights...",
        
        #check weight types
        if self.weight_type['visible_hidden'] not in self.valid_layer_types['visible_hidden']:
            print self.weight_type['visible_hidden'], "is not valid layer type. Must be one of the following:", self.valid_layer_types['visible_hidden'], "...Exiting now"
            sys.exit()
        if self.weight_type['hidden_hidden'] not in self.valid_layer_types['hidden_hidden']:
            print self.weight_type['hidden_hidden'], "is not valid layer type. Must be one of the following:", self.valid_layer_types['hidden_hidden'], "...Exiting now"
            sys.exit()
        if self.weight_type['hidden_output'] not in self.valid_layer_types['hidden_output']:
            print self.weight_type['hidden_output'], "is not valid layer type. Must be one of the following:", self.valid_layer_types['hidden_output'], "...Exiting now"
            sys.exit()
        
        #check biases
        if self.bias['visible'].shape[1] != self.weights['visible_hidden'].shape[0]:
            print "Number of visible bias dimensions: ", self.bias['visible'].shape[1],
            print " of layer visible does not equal visible weight dimensions ", self.weights['visible_hidden'].shape[0], "... Exiting now"
            sys.exit()
            
        if self.bias['output'].shape[1] != self.weights['hidden_output'].shape[1]:
            print "Number of visible bias dimensions: ", self.bias['visible'].shape[1],
            print " of layer visible does not equal output weight dimensions ", self.weights['hidden_output'].shape[1], "... Exiting now"
            sys.exit()
        
        if self.bias['hidden'].shape[1] != self.weights['visible_hidden'].shape[1]:
            print "Number of hidden bias dimensions: ", self.bias['hidden'].shape[1],
            print " of layer 0 does not equal hidden weight dimensions ", self.weights['visible_hidden'].shape[1], " of visible_hidden layer ... Exiting now"
            sys.exit()
        if self.bias['hidden'].shape[1] != self.weights['hidden_output'].shape[0]:
            print "Number of hidden bias dimensions: ", self.bias['hidden'].shape[1],
            print " of layer 0 does not equal hidden weight dimensions ", self.weights['hidden_output'].shape[0], "of hidden_output layer... Exiting now"
            sys.exit()
        if self.bias['hidden'].shape[1] != self.weights['hidden_hidden'].shape[0]:
            print "Number of hidden bias dimensions: ", self.bias['hidden'].shape[1],
            print " of layer 0 does not equal input weight dimensions ", self.weights['hidden_hidden'].shape[0], " of hidden_hidden layer... Exiting now"
            sys.exit()
        if self.bias['hidden'].shape[1] != self.weights['hidden_hidden'].shape[1]:
            print "Number of hidden bias dimensions: ", self.bias['hidden'].shape[1],
            print " of layer 0 does not equal output weight dimensions ", self.weights['hidden_hidden'].shape[1], " hidden_hidden layer... Exiting now"
            sys.exit()
        if self.bias['hidden'].shape[1] != self.init_hiddens.shape[1]:
            print "dimensionality of hidden bias", self.bias['hidden'].shape[1], "and the initial hiddens", self.init_hiddens.shape[1], "do not match. Exiting now."
            sys.exit()
            
        #check weights
        if self.weights['visible_hidden'].shape[1] != self.weights['hidden_hidden'].shape[0]:
            print "Dimensionality of visible_hidden", self.weights['visible_hidden'].shape, "does not match dimensionality of hidden_hidden", "\b:",self.weights['hidden_hidden'].shape
            print "The second dimension of visible_hidden must equal the first dimension of hidden_hidden layer"
            sys.exit()
        
        if self.weights['hidden_hidden'].shape[1] != self.weights['hidden_hidden'].shape[0]:
            print "Dimensionality of hidden_hidden", self.weights['hidden_hidden'].shape, "is not square, which it must be. Exiting now..."
            sys.exit()
        
        if self.weights['hidden_hidden'].shape[1] != self.weights['hidden_output'].shape[0]:
            print "Dimensionality of hidden_hidden", self.weights['visible_hidden'].shape, "does not match dimensionality of hidden_output", "\b:",self.weights['hidden_hidden'].shape
            print "The second dimension of hidden_hidden must equal the first dimension of hidden_output layer"
            sys.exit()
        if self.weights['hidden_hidden'].shape[1] != self.init_hiddens.shape[1]:
            print "dimensionality of hidden_hidden weights", self.weights['hidden_hidden'].shape[1], "and the initial hiddens", self.init_hiddens.shape[1], "do not match. Exiting now."
            sys.exit() 
        
        if verbose:
            print "seems copacetic"
    def write_weights(self, output_name): #completed
        weight_dict = dict()
        weight_dict['bias_visible'] = self.bias['visible']
        weight_dict['bias_hidden'] = self.bias['hidden']
        weight_dict['bias_output'] = self.bias['output']
        
        weight_dict['bias_visible_hidden'] = self.bias['visible_hidden']
        weight_dict['bias_hidden_hidden'] = self.bias['hidden_hidden']
        weight_dict['bias_hidden_output'] = self.bias['hidden_output']
        weight_dict['init_hiddens'] = self.init_hiddens
        try:
            sp.savemat(output_name, weight_dict, oned_as='column')
        except IOError:
            print "Unable to save ", self.output_name, "... Exiting now"
            sys.exit()
        else:
            print output_name, "successfully saved"
            del weight_dict
    def __neg__(self):
        nn_output = copy.deepcopy(self)
        for key in self.bias_keys:
            nn_output.bias[key] = -self.bias[key]
        for key in self.weights_keys:
            nn_output.weights[key] = -self.weights[key]
        nn_output.init_hiddens = -self.init_hiddens
        return nn_output
    def __add__(self,addend):
        nn_output = copy.deepcopy(self)
        if type(addend) is Recurrent_Neural_Network_Weight:
            if self.get_architecture() != addend.get_architecture():
                print "Neural net models do not match... Exiting now"
                sys.exit()
            
            for key in self.bias_keys:
                nn_output.bias[key] = self.bias[key] + addend.bias[key]
            for key in self.weights_keys:
                nn_output.weights[key] = self.weights[key] + addend.weights[key]
            nn_output.init_hiddens = self.init_hiddens + addend.init_hiddens
            return nn_output
        #otherwise type is scalar
        addend = float(addend)
        for key in self.bias_keys:
            nn_output.bias[key] = self.bias[key] + addend
        for key in self.weights_keys:
            nn_output.weights[key] = self.weights[key] + addend
        nn_output.init_hiddens = self.init_hiddens + addend
        return nn_output
        
    def __sub__(self,subtrahend):
        nn_output = copy.deepcopy(self)
        if type(subtrahend) is Recurrent_Neural_Network_Weight:
            if self.get_architecture() != subtrahend.get_architecture():
                print "Neural net models do not match... Exiting now"
                sys.exit()
            
            for key in self.bias_keys:
                nn_output.bias[key] = self.bias[key] - subtrahend.bias[key]
            for key in self.weights_keys:
                nn_output.weights[key] = self.weights[key] - subtrahend.weights[key]
            nn_output.init_hiddens = self.init_hiddens - subtrahend.init_hiddens
            return nn_output
        #otherwise type is scalar
        subtrahend = float(subtrahend)
        for key in self.bias_keys:
            nn_output.bias[key] = self.bias[key] - subtrahend
        for key in self.weights_keys:
            nn_output.weights[key] = self.weights[key] - subtrahend
        nn_output.init_hiddens = self.init_hiddens - subtrahend
        return nn_output
    def __mul__(self, multiplier):
        #if type(scalar) is not float and type(scalar) is not int:
        #    print "__mul__ must be by a float or int. Instead it is type", type(scalar), "Exiting now"
        #    sys.exit()
        nn_output = copy.deepcopy(self)
        if type(multiplier) is Recurrent_Neural_Network_Weight:
            for key in self.bias_keys:
                nn_output.bias[key] = self.bias[key] * multiplier.bias[key]
            for key in self.weights_keys:
                nn_output.weights[key] = self.weights[key] * multiplier.weights[key]
            nn_output.init_hiddens = self.init_hiddens * multiplier.init_hiddens
            return nn_output
        #otherwise scalar type
        multiplier = float(multiplier)
        
        for key in self.bias_keys:
            nn_output.bias[key] = self.bias[key] * multiplier
        for key in self.weights_keys:
            nn_output.weights[key] = self.weights[key] * multiplier
        nn_output.init_hiddens = self.init_hiddens * multiplier
        return nn_output
    def __div__(self, divisor):
        #if type(scalar) is not float and type(scalar) is not int:
        #    print "Divide must be by a float or int. Instead it is type", type(scalar), "Exiting now"
        #    sys.exit()
        nn_output = copy.deepcopy(self)
        if type(divisor) is Recurrent_Neural_Network_Weight:
            for key in self.bias_keys:
                nn_output.bias[key] = self.bias[key] / divisor.bias[key]
            for key in self.weights_keys:
                nn_output.weights[key] = self.weights[key] / divisor.weights[key]
            nn_output.init_hiddens = self.init_hiddens / divisor.init_hiddens
            return nn_output
        #otherwise scalar type
        divisor = float(divisor)
        
        for key in self.bias_keys:
            nn_output.bias[key] = self.bias[key] / divisor
        for key in self.weights_keys:
            nn_output.weights[key] = self.weights[key] / divisor
        nn_output.init_hiddens = self.init_hiddens / divisor
        return nn_output
    def __iadd__(self, nn_weight2):
        if type(nn_weight2) is not Recurrent_Neural_Network_Weight:
            print "argument must be of type Recurrent_Neural_Network_Weight... instead of type", type(nn_weight2), "Exiting now..."
            sys.exit()
        if self.get_architecture() != nn_weight2.get_architecture():
            print "Neural net models do not match... Exiting now"
            sys.exit()

        for key in self.bias_keys:
            self.bias[key] += nn_weight2.bias[key]
        for key in self.weights_keys:
            self.weights[key] += nn_weight2.weights[key]
        self.init_hiddens += nn_weight2.init_hiddens
        return self
    def __isub__(self, nn_weight2):
        if type(nn_weight2) is not Recurrent_Neural_Network_Weight:
            print "argument must be of type Recurrent_Neural_Network_Weight... instead of type", type(nn_weight2), "Exiting now..."
            sys.exit()
        if self.get_architecture() != nn_weight2.get_architecture():
            print "Neural net models do not match... Exiting now"
            sys.exit()

        for key in self.bias_keys:
            self.bias[key] -= nn_weight2.bias[key]
        for key in self.weights_keys:
            self.weights[key] -= nn_weight2.weights[key]
        self.init_hiddens -= nn_weight2.init_hiddens
        return self
    def __imul__(self, scalar):
        #if type(scalar) is not float and type(scalar) is not int:
        #    print "__imul__ must be by a float or int. Instead it is type", type(scalar), "Exiting now"
        #    sys.exit()
        scalar = float(scalar)
        for key in self.bias_keys:
            self.bias[key] *= scalar
        for key in self.weights_keys:
            self.weights[key] *= scalar
        self.init_hiddens *= scalar
        return self
    def __idiv__(self, scalar):
        scalar = float(scalar)
        for key in self.bias_keys:
            self.bias[key] /= scalar
        for key in self.weights_keys:
            self.weights[key] /= scalar
        self.init_hiddens /= scalar
        return self
    def __pow__(self, scalar):
        if scalar == 2:
            return self * self
        scalar = float(scalar)
        nn_output = copy.deepcopy(self)
        for key in self.bias_keys:
            nn_output.bias[key] = self.bias[key] ** scalar
        for key in self.weights_keys:
            nn_output.weights[key] = self.weights[key] ** scalar
        nn_output.init_hiddens = nn_output.init_hiddens ** scalar
        return nn_output
    def __copy__(self):
        return Recurrent_Neural_Network_Weight(self.init_hiddens, self.weights, self.bias, self.weight_type)
    def __deepcopy__(self, memo):
        return Recurrent_Neural_Network_Weight(copy.deepcopy(self.init_hiddens, memo), copy.deepcopy(self.weights,memo), 
                                     copy.deepcopy(self.bias,memo), copy.deepcopy(self.weight_type,memo))
        


class Recurrent_Neural_Network(object, Vector_Math):
    """features are stored in format max_seq_len x nseq x nvis where n_max_obs is the maximum number of observations per sequence
    and nseq is the number of sequences
    weights are stored as nvis x nhid at feature level
    biases are stored as 1 x nhid
    rbm_type is either rbm_gaussian_bernoulli, rbm_bernoulli_bernoulli, logistic"""
    def __init__(self, config_dictionary): #completed
        """variables for Neural Network: feature_file_name(read from)
        required_variables - required variables for running system
        all_variables - all valid variables for each type"""
        self.feature_file_name = self.default_variable_define(config_dictionary, 'feature_file_name', arg_type='string')
        self.features, self.feature_sequence_lens = self.read_feature_file()
        self.model = Recurrent_Neural_Network_Weight()
        self.output_name = self.default_variable_define(config_dictionary, 'output_name', arg_type='string')
        
        self.required_variables = dict()
        self.all_variables = dict()
        self.required_variables['train'] = ['mode', 'feature_file_name', 'output_name']
        self.all_variables['train'] = self.required_variables['train'] + ['label_file_name', 'num_hiddens', 'weight_matrix_name', 
                               'initial_weight_max', 'initial_weight_min', 'initial_bias_max', 'initial_bias_min', 'save_each_epoch',
                               'do_pretrain', 'pretrain_method', 'pretrain_iterations', 
                               'pretrain_learning_rate', 'pretrain_batch_size',
                               'do_backprop', 'backprop_method', 'backprop_batch_size', 'l2_regularization_const',
                               'num_epochs', 'num_line_searches', 'armijo_const', 'wolfe_const',
                               'steepest_learning_rate',
                               'conjugate_max_iterations', 'conjugate_const_type',
                               'truncated_newton_num_cg_epochs', 'truncated_newton_init_damping_factor',
                               'krylov_num_directions', 'krylov_num_batch_splits', 'krylov_num_bfgs_epochs', 'second_order_matrix',
                               'krylov_use_hessian_preconditioner', 'krylov_eigenvalue_floor_const', 
                               'fisher_preconditioner_floor_val', 'use_fisher_preconditioner']
        self.required_variables['test'] =  ['mode', 'feature_file_name', 'weight_matrix_name', 'output_name']
        self.all_variables['test'] =  self.required_variables['test'] + ['label_file_name']
    def dump_config_vals(self):
        no_attr_key = list()
        print "********************************************************************************"
        print "Neural Network configuration is as follows:"
        
        for key in self.all_variables[self.mode]:
            if hasattr(self,key):
                print key, "=", eval('self.' + key)
            else:
                no_attr_key.append(key)
                
        print "********************************************************************************"
        print "Undefined keys are as follows:"
        for key in no_attr_key:
            print key, "not set"
        print "********************************************************************************"
    def default_variable_define(self,config_dictionary,config_key, arg_type='string', 
                                default_value=None, error_string=None, exit_if_no_default=True,
                                acceptable_values=None):
        #arg_type is either int, float, string, int_comma_string, float_comma_string, boolean
        try:
            if arg_type == 'int_comma_string':
                return self.read_config_comma_string(config_dictionary[config_key], needs_int=True)
            elif arg_type == 'float_comma_string':
                return self.read_config_comma_string(config_dictionary[config_key], needs_int=False)
            elif arg_type == 'int':
                return int(config_dictionary[config_key])
            elif arg_type == 'float':
                return float(config_dictionary[config_key])
            elif arg_type == 'string':
                return config_dictionary[config_key]
            elif arg_type == 'boolean':
                if config_dictionary[config_key] == 'False' or config_dictionary[config_key] == '0' or config_dictionary[config_key] == 'F':
                    return False
                elif config_dictionary[config_key] == 'True' or config_dictionary[config_key] == '1' or config_dictionary[config_key] == 'T':
                    return True
                else:
                    print config_dictionary[config_key], "is not valid for boolean type... Acceptable values are True, False, 1, 0, T, or F... Exiting now"
                    sys.exit()
            else:
                print arg_type, "is not a valid type, arg_type can be either int, float, string, int_comma_string, float_comma_string... exiting now"
                sys.exit()
        except KeyError:
            if error_string != None:
                print error_string
            else:
                print "No", config_key, "defined,",
            if default_value == None and exit_if_no_default:
                print "since", config_key, "must be defined... exiting now"
                sys.exit()
            else:
                if acceptable_values != None and (default_value not in acceptable_values):
                    print default_value, "is not an acceptable input, acceptable inputs are", acceptable_values, "... Exiting now"
                    sys.exit()
                if error_string == None:
                    print "setting", config_key, "to", default_value
                return default_value
    def read_feature_file(self): #completed
        try:
            feature_data = sp.loadmat(self.feature_file_name)
            features = feature_data['features']
            sequence_len = feature_data['feature_sequence_lengths']
            sequence_len = np.reshape(sequence_len, (sequence_len.size,))
            return features, sequence_len#in MATLAB format
        except IOError:
            print "Unable to open ", self.feature_file_name, "... Exiting now"
            sys.exit()
    def read_label_file(self): #completed
        """label file is a two-column file in the form
        sent_id label_1
        sent_id label_2
        ...
        """
        try:
            label_data = sp.loadmat(self.label_file_name)['labels']
            return label_data[:,1], label_data[:,0]#in MATLAB format
        except IOError:
            print "Unable to open ", self.label_file_name, "... Exiting now"
            sys.exit()
    def read_config_comma_string(self,input_string,needs_int=False):
        output_list = []
        for elem in input_string.split(','):
            if '*' in elem:
                elem_list = elem.split('*')
                if needs_int:
                    output_list.extend([int(elem_list[1])] * int(elem_list[0]))
                else:
                    output_list.extend([float(elem_list[1])] * int(elem_list[0]))
            else:
                if needs_int:
                    output_list.append(int(elem))
                else:
                    output_list.append(float(elem))
        return output_list
    def levenshtein_string_edit_distance(self, string1, string2): #completed
        dist = dict()
        string1_len = len(string1)
        string2_len = len(string2)
        
        for idx in range(-1,string1_len+1):
            dist[(idx, -1)] = idx + 1
        for idx in range(-1,string2_len+1):
            dist[(-1, idx)] = idx + 1
            
        for idx1 in range(string1_len):
            for idx2 in range(string2_len):
                if string1[idx1] == string2[idx2]:
                    cost = 0
                else:
                    cost = 1
                dist[(idx1,idx2)] = min(
                           dist[(idx1-1,idx2)] + 1, # deletion
                           dist[(idx1,idx2-1)] + 1, # insertion
                           dist[(idx1-1,idx2-1)] + cost, # substitution
                           )
                if idx1 and idx2 and string1[idx1]==string2[idx2-1] and string1[idx1-1] == string2[idx2]:
                    dist[(idx1,idx2)] = min (dist[(idx1,idx2)], dist[idx1-2,idx2-2] + cost) # transposition
        return dist[(string1_len-1, string2_len-1)]    
    def check_keys(self, config_dictionary): #completed
        print "Checking config keys...",
        exit_flag = False
        
        config_dictionary_keys = config_dictionary.keys()
        
        if self.mode == 'train':
            correct_mode = 'train'
            incorrect_mode = 'test'
        elif self.mode == 'test':
            correct_mode = 'test'
            incorrect_mode = 'train'
            
        for req_var in self.required_variables[correct_mode]:
            if req_var not in config_dictionary_keys:
                print req_var, "needs to be set for", correct_mode, "but is not."
                if exit_flag == False:
                    print "Because of above error, will exit after checking rest of keys"
                    exit_flag = True
        
        for var in config_dictionary_keys:
            if var not in self.all_variables[correct_mode]:
                print var, "in the config file given is not a valid key for", correct_mode
                if var in self.all_variables[incorrect_mode]:
                    print "but", var, "is a valid key for", incorrect_mode, "so either the mode or key is incorrect"
                else:
                    string_distances = np.array([self.levenshtein_string_edit_distance(var, string2) for string2 in self.all_variables[correct_mode]])
                    print "perhaps you meant ***", self.all_variables[correct_mode][np.argmin(string_distances)], "\b*** (levenshtein string edit distance", np.min(string_distances), "\b) instead of ***", var, "\b***?"
                if exit_flag == False:
                    print "Because of above error, will exit after checking rest of keys"
                    exit_flag = True
        
        if exit_flag:
            print "Exiting now"
            sys.exit()
        else:
            print "seems copacetic"
    def check_labels(self): #want to prune non-contiguous labels, might be expensive
        #TODO: check sentids to make sure seqences are good
        print "Checking labels..."
        if len(self.labels.shape) != 1 :
            print "labels need to be in (n_samples) or (n_samples,2) format and the shape of labels is ", self.labels.shape, "... Exiting now"
            sys.exit()
        if self.labels.size != sum(self.feature_sequence_lens):
            print "Number of examples in feature file: ", sum(self.feature_sequence_lens), " does not equal size of label file, ", self.labels.size, "... Exiting now"
            sys.exit()
        if  [i for i in np.unique(self.labels)] != range(np.max(self.labels)+1):
            print "Labels need to be in the form 0,1,2,....,n,... Exiting now"
            sys.exit()
        label_counts = np.bincount(np.ravel(self.labels)) #[self.labels.count(x) for x in range(np.max(self.labels)+1)]
        print "distribution of labels is:"
        for x in range(len(label_counts)):
            print "#", x, "\b's:", label_counts[x]            
        print "labels seem copacetic"
    def forward_layer(self, inputs, weights, biases, weight_type, prev_hiddens = None, hidden_hidden_weights = None): #completed
        if weight_type == 'logistic':
            return self.softmax(self.weight_matrix_multiply(inputs, weights, biases))
        #TODO: make the tanh rigorous
        elif weight_type == 'rbm_gaussian_bernoulli' or weight_type == 'rbm_bernoulli_bernoulli':
            return np.tanh(self.weight_matrix_multiply(inputs, weights, biases) + np.dot(prev_hiddens, hidden_hidden_weights))
        #added to test finite differences calculation for pearlmutter forward pass
        elif weight_type == 'linear': #only used for the logistic layer
            return self.weight_matrix_multiply(inputs, weights, biases)
        else:
            print "weight_type", weight_type, "is not a valid layer type.",
            print "Valid layer types are", self.model.valid_layer_types,"Exiting now..."
            sys.exit()
#    def forward_pass_linear(self, inputs, verbose=True, model=None):
#        #to test finite differences calculation for pearlmutter forward pass, just like forward pass, except it spits linear outputs
#        if model == None:
#            model = self.model
#        architecture = self.model.get_architecture()
#        max_sequence_observations = inputs.shape[0]
#        num_hiddens = architecture[1]
#        num_sequences = inputs.shape[2]
#        num_outs = architecture[2]
#        hiddens = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
#        outputs = np.zeros((max_sequence_observations, num_sequences, num_outs))
#        
#        #propagate hiddens
#        hiddens[0,:,:] = self.forward_layer(inputs[0,:,:], self.model.weights['visible_hidden'], self.model.bias['hidden'], 
#                                            self.model.weight_type['visible_hidden'], self.model.init_hiddens, 
#                                            self.model.weights['hidden_hidden'])
#        outputs[0,:,:] = self.forward_layer(hiddens[0,:,:], self.model.weights['hidden_output'], self.model.bias['output'], 
#                                            self.model.weight_type['hidden_output'])
#        for sequence_index in range(1, max_sequence_observations):
#            sequence_input = inputs[sequence_index,:,:]
#            hiddens[sequence_index,:,:] = self.forward_layer(sequence_input, self.model.weights['visible_hidden'], self.model.bias['hidden'], 
#                                                             self.model.weight_type['visible_hidden'], hiddens[sequence_index-1,:,:], 
#                                                             self.model.weights['hidden_hidden'])
#            outputs[sequence_index,:,:] = self.forward_layer(hiddens[sequence_index,:,:], self.model.weights['hidden_output'], self.model.bias['output'], 
#                                                             'linear')
#            #find the observations where the sequence has ended, 
#            #and then zero out hiddens and outputs, so nothing horrible happens during backprop, etc.
#            zero_input = np.where(self.feature_sequence_lens >= sequence_index)
#            hiddens[sequence_index,:,zero_input] = 0.0
#            outputs[sequence_index,:,zero_input] = 0.0
#
#        del hiddens
#        return outputs
    def forward_pass(self, inputs, model=None, return_hiddens=False, linear_output=False): #completed
        """forward pass each layer starting with feature level
        inputs in the form n_max_obs x n_seq x n_vis"""
        if model == None:
            model = self.model
        architecture = self.model.get_architecture()
        max_sequence_observations = inputs.shape[0]
        num_sequences = inputs.shape[1]
        num_hiddens = architecture[1]
        num_outs = architecture[2]
        hiddens = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
        outputs = np.zeros((max_sequence_observations, num_sequences, num_outs))
        #propagate hiddens
        hiddens[0,:,:] = self.forward_layer(inputs[0,:,:], model.weights['visible_hidden'], model.bias['hidden'], 
                                            model.weight_type['visible_hidden'], model.init_hiddens, 
                                            model.weights['hidden_hidden'])
        if linear_output:
            outputs[0,:,:] = self.forward_layer(hiddens[0,:,:], model.weights['hidden_output'], model.bias['output'], 
                                                'linear')
        else:
            outputs[0,:,:] = self.forward_layer(hiddens[0,:,:], model.weights['hidden_output'], model.bias['output'], 
                                                model.weight_type['hidden_output'])
        for sequence_index in range(1, max_sequence_observations):
            sequence_input = inputs[sequence_index,:,:]
            hiddens[sequence_index,:,:] = self.forward_layer(sequence_input, model.weights['visible_hidden'], model.bias['hidden'], 
                                                             model.weight_type['visible_hidden'], hiddens[sequence_index-1,:,:], 
                                                             model.weights['hidden_hidden'])
            if linear_output:
                outputs[sequence_index,:,:] = self.forward_layer(hiddens[sequence_index,:,:], model.weights['hidden_output'], model.bias['output'], 
                                                             'linear')
            else:
                outputs[sequence_index,:,:] = self.forward_layer(hiddens[sequence_index,:,:], model.weights['hidden_output'], model.bias['output'], 
                                                                 model.weight_type['hidden_output'])
            #find the observations where the sequence has ended, 
            #and then zero out hiddens and outputs, so nothing horrible happens during backprop, etc.
            zero_input = np.where(self.feature_sequence_lens <= sequence_index)
            hiddens[sequence_index,zero_input,:] = 0.0
            outputs[sequence_index,zero_input,:] = 0.0
        if return_hiddens:
            return outputs, hiddens
        else:
            del hiddens
            return outputs
    def flatten_output(self, output, feature_sequence_lens=None):
        """outputs in the form of max_obs_seq x n_seq x n_outs  get converted to form
        n_data x n_outs, so we can calculate classification accuracy and cross-entropy
        """
        if feature_sequence_lens == None:
            feature_sequence_lens = self.feature_sequence_lens
        num_outs = output.shape[2]
#        num_seq = output.shape[1]
        flat_output = np.zeros((self.batch_size(feature_sequence_lens), num_outs))
        cur_index = 0
        for seq_index, num_obs in enumerate(feature_sequence_lens):
            flat_output[cur_index:cur_index+num_obs, :] = copy.deepcopy(output[:num_obs, seq_index, :])
            cur_index += num_obs
        
        return flat_output
    def calculate_cross_entropy(self, output, labels): #completed, expensive, should be compiled
        """calculates cross_entropy whether or not labels or outputs are flat
        """
        return -np.sum(np.log(np.clip(output, a_min=1E-12, a_max=1.0)) * labels) 
    def calculate_classification_accuracy(self, flat_output, labels): #completed, possibly expensive
        prediction = flat_output.argmax(axis=1).reshape(labels.shape)
        classification_accuracy = sum(prediction == labels) / float(labels.size)
        return classification_accuracy[0]
    

class RNN_Tester(Recurrent_Neural_Network): #completed
    def __init__(self, config_dictionary): #completed
        """runs DNN tester soup to nuts.
        variables are
        feature_file_name - name of feature file to load from
        weight_matrix_name - initial weight matrix to load
        output_name - output predictions
        label_file_name - label file to check accuracy
        required are feature_file_name, weight_matrix_name, and output_name"""
        self.mode = 'test'
        super(RNN_Tester,self).__init__(config_dictionary)
        self.check_keys(config_dictionary)
        
        self.weight_matrix_name = self.default_variable_define(config_dictionary, 'weight_matrix_name', arg_type='string')
        self.model.open_weights(self.weight_matrix_name)
        self.label_file_name = self.default_variable_define(config_dictionary, 'label_file_name', arg_type='string',error_string="No label_file_name defined, just running forward pass",exit_if_no_default=False)
        if self.label_file_name != None:
            self.labels, self.labels_sent_id = self.read_label_file()
            self.check_labels()
        else:
            del self.label_file_name
        self.dump_config_vals()
        self.classify()
        self.write_posterior_prob_file()
    def classify(self): #completed
        self.posterior_probs = self.forward_pass(self.features)
        self.flat_posterior_probs = self.flatten_output(self.posterior_probs)
        try:
            avg_cross_entropy = self.calculate_cross_entropy(self.flat_posterior_probs, self.labels) / self.labels.size
            print "Average cross-entropy is", avg_cross_entropy
            print "Classification accuracy is", self.calculate_classification_accuracy(self.flat_posterior_probs, self.labels) * 100, "\b%"
        except AttributeError:
            print "no labels given, so skipping classification statistics"    
    def write_posterior_prob_file(self): #completed
        try:
            print "Writing to", self.output_name
            sp.savemat(self.output_name,{'targets' : self.posterior_probs, 'sequence_lengths' : self.feature_sequence_lens}, oned_as='column') #output name should have .mat extension
        except IOError:
            print "Unable to write to ", self.output_name, "... Exiting now"
            sys.exit()

class RNN_Trainer(Recurrent_Neural_Network):
    def __init__(self,config_dictionary): #completed
        """variables in NN_trainer object are:
        mode (set to 'train')
        feature_file_name - inherited from Neural_Network class, name of feature file (in .mat format with variable 'features' in it) to read from
        features - inherited from Neural_Network class, features
        label_file_name - name of label file (in .mat format with variable 'labels' in it) to read from
        labels - labels for backprop
        architecture - specified by n_hid, n_hid, ..., n_hid. # of feature dimensions and # of classes need not be specified
        weight_matrix_name - initial weight matrix, if specified, if not, will initialize from random
        initial_weight_max - needed if initial weight matrix not loaded
        initial_weight_min - needed if initial weight matrix not loaded
        initial_bias_max - needed if initial weight matrix not loaded
        initial_bias_min - needed if initial weight matrix not loaded
        do_pretrain - set to 1 or 0 (probably should change to boolean values)
        pretrain_method - not yet implemented, will either be 'mean_field' or 'sampling'
        pretrain_iterations - # of iterations per RBM. Must be equal to the number of hidden layers
        pretrain_learning_rate - learning rate for each epoch of pretrain. must be equal to # hidden layers * sum(pretrain_iterations)
        pretrain_batch_size - batch size for pretraining
        do_backprop - do backpropagation (set to either 0 or 1, probably should be changed to boolean value)
        backprop_method - either 'steepest_descent', 'conjugate_gradient', or '2nd_order', latter two not yet implemented
        l2_regularization_constant - strength of l2 (weight decay) regularization
        steepest_learning_rate - learning rate for steepest_descent backprop
        backprop_batch_size - batch size for backprop
        output_name - name of weight file to store to.
        ********************************************************************************
         At bare minimum, you'll need these variables set to train
         feature_file_name
         output_name
         this will run logistic regression using steepest descent, which is a bad idea"""
        
        #Raise error if we encounter under/overflow during training, because this is bad... code should handle this gracefully
        old_settings = np.seterr(over='raise',under='raise',invalid='raise')
        
        self.mode = 'train'
        super(RNN_Trainer,self).__init__(config_dictionary)
        self.num_training_examples = self.batch_size(self.feature_sequence_lens)
        self.num_sequences = self.features.shape[1]
        self.check_keys(config_dictionary)
        #read label file
        self.label_file_name = self.default_variable_define(config_dictionary, 'label_file_name', arg_type='string', error_string="No label_file_name defined, can only do pretraining",exit_if_no_default=False)
        if self.label_file_name != None:
            self.labels, self.labels_sent_id = self.read_label_file()
            self.check_labels()
            self.unflattened_labels = self.unflatten_labels(self.labels, self.labels_sent_id) 
        else:
            del self.label_file_name        

        #initialize weights
        self.weight_matrix_name = self.default_variable_define(config_dictionary, 'weight_matrix_name', exit_if_no_default=False)

        if self.weight_matrix_name != None:
            print "Since weight_matrix_name is defined, ignoring possible value of hiddens_structure"
            self.model.open_weights(self.weight_matrix_name)
        else: #initialize model
            del self.weight_matrix_name
            
            self.num_hiddens = self.default_variable_define(config_dictionary, 'num_hiddens', arg_type='int', exit_if_no_default=True)
            architecture = [self.features.shape[2], self.num_hiddens]
            if hasattr(self, 'labels'):
                architecture.append(np.max(self.labels)+1) #will have to change later if I have soft weights
                
            self.initial_weight_max = self.default_variable_define(config_dictionary, 'initial_weight_max', arg_type='float', default_value=0.1)
            self.initial_weight_min = self.default_variable_define(config_dictionary, 'initial_weight_min', arg_type='float', default_value=-0.1)
            self.initial_bias_max = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=-2.2)
            self.initial_bias_min = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=-2.4)
            self.model.init_random_weights(architecture, self.initial_bias_max, self.initial_bias_min, 
                                           self.initial_weight_min, self.initial_weight_max)
            del architecture #we have it in the model
        #
        
        self.save_each_epoch = self.default_variable_define(config_dictionary, 'save_each_epoch', arg_type='boolean', default_value=False)
        #pretraining configuration
        self.do_pretrain = self.default_variable_define(config_dictionary, 'do_pretrain', default_value=False, arg_type='boolean')
        if self.do_pretrain:
            self.pretrain_method = self.default_variable_define(config_dictionary, 'pretrain_method', default_value='mean_field', acceptable_values=['mean_field', 'sampling'])
            self.pretrain_iterations = self.default_variable_define(config_dictionary, 'pretrain_iterations', default_value=[5] * len(self.hiddens_structure), 
                                                                    error_string="No pretrain_iterations defined, setting pretrain_iterations to default 5 per layer", 
                                                                    arg_type='int_comma_string')

            weight_last_layer = ''.join([str(self.model.num_layers-1), str(self.model.num_layers)])
            if self.model.weight_type[weight_last_layer] == 'logistic' and (len(self.pretrain_iterations) != self.model.num_layers - 1):
                print "given layer type", self.model.weight_type[weight_last_layer], "pretraining iterations length should be", self.model.num_layers-1, "but pretraining_iterations is length ", len(self.pretrain_iterations), "... Exiting now"
                sys.exit()
            elif self.model.weight_type[weight_last_layer] != 'logistic' and (len(self.pretrain_iterations) != self.model.num_layers):
                print "given layer type", self.model.weight_type[weight_last_layer], "pretraining iterations length should be", self.model.num_layers, "but pretraining_iterations is length ", len(self.pretrain_iterations), "... Exiting now"
                sys.exit()
            self.pretrain_learning_rate = self.default_variable_define(config_dictionary, 'pretrain_learning_rate', default_value=[0.01] * sum(self.pretrain_iterations), 
                                                                       error_string="No pretrain_learning_rate defined, setting pretrain_learning_rate to default 0.01 per iteration", 
                                                                       arg_type='float_comma_string')
            if len(self.pretrain_learning_rate) != sum(self.pretrain_iterations):
                print "pretraining learning rate should have ", sum(self.pretrain_iterations), " learning rate iterations but only has ", len(self.pretrain_learning_rate), "... Exiting now"
                sys.exit()
            self.pretrain_batch_size = self.default_variable_define(config_dictionary, 'pretrain_batch_size', default_value=256, arg_type='int')
                    
        #backprop configuration
        self.do_backprop = self.default_variable_define(config_dictionary, 'do_backprop', default_value=True, arg_type='boolean')
        if self.do_backprop:
            if not hasattr(self, 'labels'):
                print "No labels found... cannot do backprop... Exiting now"
                sys.exit()
            self.backprop_method = self.default_variable_define(config_dictionary, 'backprop_method', default_value='steepest_descent', 
                                                                acceptable_values=['steepest_descent', 'krylov_subspace', 'truncated_newton'])
            self.backprop_batch_size = self.default_variable_define(config_dictionary, 'backprop_batch_size', default_value=16, arg_type='int')
            self.l2_regularization_const = self.default_variable_define(config_dictionary, 'l2_regularization_const', arg_type='float', default_value=0.0, exit_if_no_default=False)
            
            if self.backprop_method == 'steepest_descent':
                self.steepest_learning_rate = self.default_variable_define(config_dictionary, 'steepest_learning_rate', default_value=[0.08, 0.04, 0.02, 0.01], arg_type='float_comma_string')
            else: #second order methods
                self.num_epochs = self.default_variable_define(config_dictionary, 'num_epochs', default_value=20, arg_type='int')
                self.use_fisher_preconditioner = self.default_variable_define(config_dictionary, 'use_fisher_preconditioner', arg_type='boolean', default_value=False)
                self.second_order_matrix = self.default_variable_define(config_dictionary, 'second_order_matrix', arg_type='string', default_value='gauss-newton', 
                                                                        acceptable_values=['gauss-newton', 'hessian', 'fisher'])
                self.structural_damping_const = self.default_variable_define(config_dictionary, 'structural_damping_const', arg_type='float', default_value=0.0, exit_if_no_default=False)
                if self.use_fisher_preconditioner:
                    self.fisher_preconditioner_floor_val = self.default_variable_define(config_dictionary, 'fisher_preconditioner_floor_val', arg_type='float', default_value=1E-4)
                if self.backprop_method == 'krylov_subspace':
                    self.krylov_num_directions = self.default_variable_define(config_dictionary, 'krylov_num_directions', arg_type='int', default_value=20, 
                                                                              acceptable_values=range(2,2000))
                    self.krylov_num_batch_splits = self.default_variable_define(config_dictionary, 'krylov_num_batch_splits', arg_type='int', default_value=self.krylov_num_directions, 
                                                                                acceptable_values=range(2,2000))
                    self.krylov_num_bfgs_epochs = self.default_variable_define(config_dictionary, 'krylov_num_bfgs_epochs', arg_type='int', default_value=self.krylov_num_directions)
                    self.krylov_use_hessian_preconditioner = self.default_variable_define(config_dictionary, 'krylov_use_hessian_preconditioner', arg_type='boolean', default_value=True)
                    if self.krylov_use_hessian_preconditioner:
                        self.krylov_eigenvalue_floor_const = self.default_variable_define(config_dictionary, 'krylov_eigenvalue_floor_const', arg_type='float', default_value=1E-4)
                    
                    self.num_line_searches = self.default_variable_define(config_dictionary, 'num_line_searches', default_value=20, arg_type='int')
                    self.armijo_const = self.default_variable_define(config_dictionary, 'armijo_const', arg_type='float', default_value=0.0001)
                    self.wolfe_const = self.default_variable_define(config_dictionary, 'wolfe_const', arg_type='float', default_value=0.9)
                elif self.backprop_method == 'truncated_newton':
                    self.truncated_newton_num_cg_epochs = self.default_variable_define(config_dictionary, 'truncated_newton_num_cg_epochs', arg_type='int', default_value=20)
                    self.truncated_newton_init_damping_factor = self.default_variable_define(config_dictionary, 'truncated_newton_init_damping_factor', arg_type='float', default_value=0.1)
        self.dump_config_vals()
    def batch_size(self, feature_sequence_lens):
        return np.sum(feature_sequence_lens)
    def unflatten_labels(self, labels, sentence_ids):
        num_frames_per_sentence = np.bincount(sentence_ids)
        num_outs = len(np.unique(labels))
        max_num_frames_per_sentence = np.max(num_frames_per_sentence)
        unflattened_labels = np.zeros((max_num_frames_per_sentence, np.max(sentence_ids) + 1, num_outs)) #add one because first sentence starts at 0
        current_sentence_id = 0
        observation_index = 0
        for label, sentence_id in zip(labels,sentence_ids):
            if sentence_id != current_sentence_id:
                current_sentence_id = sentence_id
                observation_index = 0
            unflattened_labels[observation_index, sentence_id, label] = 1.0
            observation_index += 1
        return unflattened_labels
    def backward_pass(self, backward_inputs, hiddens, visibles, model=None, structural_damping_const = 0.0, hidden_deriv = None): #need to test
        
        if model == None:
            model = self.model
        output_model = Recurrent_Neural_Network_Weight()
        output_model.init_zero_weights(self.model.get_architecture(), verbose=False)
        
        n_obs = backward_inputs.shape[0]
        n_outs = backward_inputs.shape[2]
        n_hids = hiddens.shape[2]
        n_seq = backward_inputs.shape[1]
        #backward_inputs - n_obs x n_seq x n_outs
        #hiddens - n_obs  x n_seq x n_hids
        flat_outputs = np.reshape(np.transpose(backward_inputs, axes=(1,0,2)),(n_obs * n_seq,n_outs))
        flat_hids = np.reshape(np.transpose(hiddens, axes=(1,0,2)),(n_obs * n_seq,n_hids))
        #average layers in batch
        
        output_model.bias['output'][0] = np.sum(flat_outputs, axis=0)
        output_model.weights['hidden_output'] = np.dot(flat_hids.T, flat_outputs)
        
        #HERE is where the fun begins... will need to store gradient updates in the form of dL/da_i 
        #(where a is the pre-nonlinear layer for updates to the hidden hidden and input hidden weight matrices
        
        pre_nonlinearity_hiddens = np.dot(backward_inputs[n_obs-1,:,:], model.weights['hidden_output'].T) * (1 + hiddens[n_obs-1,:,:]) * (1 - hiddens[n_obs-1,:,:])
        if structural_damping_const > 0.0:
            pre_nonlinearity_hiddens += structural_damping_const * hidden_deriv[n_obs-1,:,:]
        output_model.weights['visible_hidden'] += np.dot(visibles[n_obs-1,:,:].T, pre_nonlinearity_hiddens)
        output_model.weights['hidden_hidden'] += np.dot(hiddens[n_obs-2,:,:].T, pre_nonlinearity_hiddens)
        output_model.bias['hidden'][0] += np.sum(pre_nonlinearity_hiddens, axis=0)
        for observation_index in range(1,n_obs-1)[::-1]:
            pre_nonlinearity_hiddens = ((np.dot(backward_inputs[observation_index,:,:], model.weights['hidden_output'].T) + np.dot(pre_nonlinearity_hiddens, model.weights['hidden_hidden'].T))
                                        * (1 + hiddens[observation_index,:,:]) * (1 - hiddens[observation_index,:,:]))
            if structural_damping_const > 0.0:
                pre_nonlinearity_hiddens += structural_damping_const * hidden_deriv[observation_index,:,:]
            output_model.weights['visible_hidden'] += np.dot(visibles[observation_index,:,:].T, pre_nonlinearity_hiddens)
            output_model.weights['hidden_hidden'] += np.dot(hiddens[observation_index-1,:,:].T, pre_nonlinearity_hiddens)
            output_model.bias['hidden'][0] += np.sum(pre_nonlinearity_hiddens, axis=0)
        
        pre_nonlinearity_hiddens = ((np.dot(backward_inputs[0,:,:], model.weights['hidden_output'].T) + np.dot(pre_nonlinearity_hiddens, model.weights['hidden_hidden'].T))
                                    * (1 + hiddens[0,:,:]) * (1 - hiddens[0,:,:]))
        output_model.weights['visible_hidden'] += np.dot(visibles[0,:,:].T, pre_nonlinearity_hiddens)
        output_model.weights['hidden_hidden'] += np.dot(np.tile(model.init_hiddens, (pre_nonlinearity_hiddens.shape[0],1)).T, pre_nonlinearity_hiddens)
        output_model.bias['hidden'][0] += np.sum(pre_nonlinearity_hiddens, axis=0)
        output_model.init_hiddens[0] = np.sum(np.dot(pre_nonlinearity_hiddens, model.weights['hidden_hidden'].T), axis=0)
#        pre_nonlinearity_hiddens = np.dot(backward_inputs, model.weights['hidden_output'].T) * (1 + hiddens) * (1 - hiddens)
#
#        for observation_index in range(n_obs-1)[::-1]:
#            pre_nonlinearity_hiddens[observation_index,:,:] += (np.dot(pre_nonlinearity_hiddens[observation_index+1,:,:], model.weights['hidden_hidden'].T) * 
#                                                                (1 + hiddens[observation_index,:,:]) * (1 - hiddens[observation_index,:,:]) )
#        
#        flat_pre_nonlinearity_hiddens = np.reshape(np.transpose(pre_nonlinearity_hiddens, axes=(1,0,2)),(n_obs * n_seq,n_hids))
#        
#        output_model.bias['hidden'][0] = np.sum(flat_pre_nonlinearity_hiddens, axis=0)
#        output_model.weights['hidden_hidden'] = np.dot(flat_hids.T, flat_pre_nonlinearity_hiddens)
#        
#        output_model.weights['visible_hidden'] = np.dot(flat_vis.T, flat_pre_nonlinearity_hiddens)
#        output_model.init_hiddens = np.sum(np.dot(pre_nonlinearity_hiddens[0,:,:], model.weights['hidden_hidden'].T), axis=0)
        
        return output_model
    def calculate_loss(self, inputs, labels, batch_size, model = None, l2_regularization_const = None):
        #differs from calculate_cross_entropy in that it also allows for regularization term
        if model == None:
            model = self.model
        if l2_regularization_const == None:
            l2_regularization_const = self.l2_regularization_const
        excluded_keys = {'bias':['0'], 'weights':[]}
        outputs = self.forward_pass(inputs, model = model)
        if self.l2_regularization_const == 0.0:
            return self.calculate_cross_entropy(outputs, labels)
        else:
            return self.calculate_cross_entropy(outputs, labels) + (model.norm(excluded_keys) ** 2) * l2_regularization_const / 2. * batch_size
    def calculate_classification_statistics(self, features, unflattened_labels, feature_sequence_lens, model=None):
        if model == None:
            model = self.model
        
        excluded_keys = {'bias': ['0'], 'weights': []}
        
        if self.do_backprop == False:
            classification_batch_size = 256
        else:
            classification_batch_size = max(self.backprop_batch_size, 256)
        
        batch_index = 0
        end_index = 0
        cross_entropy = 0.0
        num_correct = 0
        num_sequences = features.shape[1]
        num_examples = self.batch_size(feature_sequence_lens)
        while end_index < num_sequences: #run through the batches
            end_index = min(batch_index+classification_batch_size, num_sequences)
            output = self.forward_pass(features[:,batch_index:end_index,:], model=model)
            cross_entropy += self.calculate_cross_entropy(output, unflattened_labels[:,batch_index:end_index,:])
            
            #don't use calculate_classification_accuracy() because of possible rounding error
            prediction = output.argmax(axis=2)
            label = np.argmax(unflattened_labels[:,batch_index:end_index,:], axis=2)
            num_correct += np.sum(prediction == label) - (prediction.size - num_examples) #because of the way we handle features, where some observations are null, we want to remove those examples for calculating accuracy
            batch_index += classification_batch_size
        
        loss = cross_entropy
        if self.l2_regularization_const > 0.0:
            loss += (model.norm(excluded_keys) ** 2) * self.l2_regularization_const
        return cross_entropy, num_correct, num_examples, loss
    def backprop_steepest_descent(self):
        print "Starting backprop using steepest descent"
        
        cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.unflattened_labels, self.feature_sequence_lens, self.model)
        print "cross-entropy before steepest descent is", cross_entropy
        if self.l2_regularization_const > 0.0:
            print "regularized loss is", loss
        print "number correctly classified is", num_correct, "of", num_examples
        
        excluded_keys = {'bias':['0'], 'weights':[]}
        
        for epoch_num in range(len(self.steepest_learning_rate)):
            print "At epoch", epoch_num+1, "of", len(self.steepest_learning_rate), "with learning rate", self.steepest_learning_rate[epoch_num]
            batch_index = 0
            end_index = 0
            
            while end_index < self.num_sequences: #run through the batches
                per_done = float(batch_index)/self.num_sequences*100
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
                end_index = min(batch_index+self.backprop_batch_size,self.num_sequences)
                batch_inputs = self.features[:,batch_index:end_index,:]
                batch_labels = self.unflattened_labels[:,batch_index:end_index,:]
                batch_size = self.batch_size(self.feature_sequence_lens[batch_index:end_index])
                
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\rcalculating gradient\r"), sys.stdout.flush()
                gradient = self.calculate_gradient(batch_inputs, batch_labels, batch_size, model=self.model)
                self.model -= gradient * self.steepest_learning_rate[epoch_num]
                batch_index += self.backprop_batch_size
                
                cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.unflattened_labels, self.feature_sequence_lens, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
                if self.l2_regularization_const > 0.0:
                    print "regularized loss is", loss
                print "number correctly classified is", num_correct, "of", num_examples
                
            sys.stdout.write("\r100.0% done \r")
            sys.stdout.write("\r                                                                \r") #clear line
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))
    def backprop_truncated_newton(self):
        print "Starting backprop using truncated newton"
        
        cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.unflattened_labels, self.feature_sequence_lens, self.model)
        print "cross-entropy before truncated newton is", cross_entropy
        if self.l2_regularization_const > 0.0:
            print "regularized loss is", loss
        print "number correctly classified is", num_correct, "of", num_examples
        
        excluded_keys = {'bias':['0'], 'weights':[]} 
        damping_factor = self.truncated_newton_init_damping_factor
        preconditioner = None
        model_update = None
        for epoch_num in range(self.num_epochs):
            print "Epoch", epoch_num+1, "of", self.num_epochs
            batch_index = 0
            end_index = 0
            
            while end_index < self.num_sequences: #run through the batches
                per_done = float(batch_index)/self.num_sequences*100
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\rdamping factor is %f\r" % damping_factor), sys.stdout.flush()
                end_index = min(batch_index+self.backprop_batch_size,self.num_sequences)
                batch_inputs = self.features[:,batch_index:end_index,:]
                batch_unflattened_labels = self.unflattened_labels[:,batch_index:end_index,:]
                batch_size = self.batch_size(self.feature_sequence_lens[batch_index:end_index])
                
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\rcalculating gradient\r"), sys.stdout.flush()
                gradient = self.calculate_gradient(batch_inputs, batch_unflattened_labels, batch_size, model=self.model)
                
                old_loss = self.calculate_loss(batch_inputs, batch_unflattened_labels, batch_size, model=self.model) 
                
                if self.use_fisher_preconditioner:
                    sys.stdout.write("\r                                                                \r")
                    sys.stdout.write("calculating diagonal Fisher matrix for preconditioner"), sys.stdout.flush()
                    
                    preconditioner = self.calculate_fisher_diag_matrix(batch_inputs, batch_unflattened_labels, False, self.model, l2_regularization_const = 0.0)
                    # add regularization
                    #preconditioner = preconditioner + alpha / preconditioner.size(excluded_keys) * self.model.norm(excluded_keys) ** 2
                    preconditioner = (preconditioner + self.l2_regularization_const + damping_factor) ** (3./4.)
                    preconditioner = preconditioner.clip(preconditioner.max(excluded_keys) * self.fisher_preconditioner_floor_val, float("Inf"))
                model_update, model_vals = self.conjugate_gradient(batch_inputs, batch_unflattened_labels, batch_size, self.truncated_newton_num_cg_epochs, 
                                                                   model=self.model, damping_factor=damping_factor, preconditioner=preconditioner, 
                                                                   gradient=gradient, second_order_type=self.second_order_matrix, 
                                                                   init_search_direction=model_update, verbose = False,
                                                                    structural_damping_const = self.structural_damping_const)
                model_den = model_vals[-1] #- model_vals[0]
                
                self.model += model_update
                new_loss = self.calculate_loss(batch_inputs, batch_unflattened_labels, batch_size, model=self.model) 
                model_num = (new_loss - old_loss) / batch_size
                sys.stdout.write("\r                                                                      \r") #clear line
                print "model ratio is", model_num / model_den,
                if model_num / model_den < 0.25:
                    damping_factor *= 1.5
                elif model_num / model_den > 0.75:
                    damping_factor *= 2./3.
                batch_index += self.backprop_batch_size
                
                cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.unflattened_labels, self.feature_sequence_lens, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
                if self.l2_regularization_const > 0.0:
                    print "regularized loss is", loss
                print "number correctly classified is", num_correct, "of", num_examples
                
            sys.stdout.write("\r100.0% done \r")
            
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))
    def conjugate_gradient(self, batch_inputs, batch_unflattened_labels, batch_size, num_epochs, model = None, damping_factor = 0.0, #seems to be correct, compare with conjugate_gradient.py
                           verbose = False, preconditioner = None, gradient = None, second_order_type='gauss-newton', 
                           init_search_direction = None, structural_damping_const = 0.0):
        """minimizes function q_x(p) = \grad_x f(x)' p + 1/2 * p'Gp (where x is fixed) use linear conjugate gradient"""
        if verbose:
            print "preconditioner is", preconditioner
        excluded_keys = {'bias':['0'], 'weights':[]} 
        if model == None:
            model = self.model
        
        tolerance = 5E-4
        gap_ratio = 0.1
        min_gap = 10
        #max_test_gap = int(np.max([np.ceil(gap_ratio * num_epochs), min_gap]) + 1)
        model_vals = list()
        
        model_update = Recurrent_Neural_Network_Weight()
        model_update.init_zero_weights(model.get_architecture())
        
        outputs, hiddens = self.forward_pass(batch_inputs, model, return_hiddens=True)
        if gradient == None:
            gradient = self.calculate_gradient(batch_inputs, batch_unflattened_labels, batch_size, model = model, hiddens = hiddens, outputs = outputs)
        
        if init_search_direction == None:
            model_vals.append(0)
            residual = gradient 
        else:
            second_order_direction = self.calculate_second_order_direction(batch_inputs, batch_unflattened_labels, init_search_direction, 
                                                                           model, second_order_type=second_order_type, hiddens = hiddens,
                                                                           structural_damping_const = structural_damping_const * damping_factor)
            residual = gradient + second_order_direction
            model_val = 0.5 * init_search_direction.dot(gradient + residual, excluded_keys)
            model_vals.append(model_val) 
            model_update += init_search_direction    
            
        if verbose:
            print "model val at end of epoch is", model_vals[-1]
        
        if preconditioner != None:
            preconditioned_residual = residual / preconditioner
        else:
            preconditioned_residual = residual
        search_direction = -preconditioned_residual
        residual_dot = residual.dot(preconditioned_residual, excluded_keys)
        for epoch in range(num_epochs):
            print "\r                                                                \r", #clear line
            sys.stdout.write("\rconjugate gradient epoch %d of %d\r" % (epoch+1, num_epochs)), sys.stdout.flush()
            
            if damping_factor > 0.0:
                #TODO: check to see if ... + search_direction * damping_factor is correct with structural damping
                second_order_direction = self.calculate_second_order_direction(batch_inputs, batch_unflattened_labels, search_direction, model, second_order_type=second_order_type, hiddens = hiddens, 
                                                                               structural_damping_const = damping_factor * structural_damping_const) + search_direction * damping_factor
            else:
                second_order_direction = self.calculate_second_order_direction(batch_inputs, batch_unflattened_labels, search_direction, model, second_order_type=second_order_type, hiddens = hiddens)
                                                                            
            curvature = search_direction.dot(second_order_direction,excluded_keys)
            if curvature <= 0:
                print "curvature must be positive, but is instead", curvature, "returning current weights"
                break
            
            step_size = residual_dot / curvature
            if verbose:
                print "residual dot search direction is", residual.dot(search_direction, excluded_keys)
                print "residual dot is", residual_dot
                print "curvature is", curvature
                print "step size is", step_size
            model_update += search_direction * step_size
            
            residual += second_order_direction * step_size
            model_val = 0.5 * model_update.dot(gradient + residual, excluded_keys)
            model_vals.append(model_val)
            if verbose:
                print "model val at end of epoch is", model_vals[-1]
            test_gap = int(np.max([np.ceil(epoch * gap_ratio), min_gap]))
            if epoch > test_gap: #checking termination condition
                previous_model_val = model_vals[-test_gap]
                if (previous_model_val - model_val) / model_val <= tolerance * test_gap and previous_model_val < 0:
                    print "\r                                                                \r", #clear line
                    sys.stdout.write("\rtermination condition satisfied for conjugate gradient, returning step\r"), sys.stdout.flush()
                    break
            if preconditioner != None:
                preconditioned_residual = residual / preconditioner
            else:
                preconditioned_residual = residual
            new_residual_dot = residual.dot(preconditioned_residual, excluded_keys)
            conjugate_gradient_const = new_residual_dot / residual_dot
            search_direction = -preconditioned_residual + search_direction * conjugate_gradient_const
            residual_dot = new_residual_dot
        return model_update, model_vals         
    def calculate_gradient(self, batch_inputs, batch_unflattened_labels, batch_size, hiddens = None, outputs = None,
                           check_gradient=False, model=None, l2_regularization_const = 0.0, flat_labels=None): 
        #need to check regularization
        #calculate gradient with particular Neural Network model. If None is specified, will use current weights (i.e., self.model)
        excluded_keys = {'bias':['0'], 'weights':[]} #will have to change this later
        if model == None:
            model = self.model
        if hiddens == None or outputs == None:
            outputs, hiddens = self.forward_pass(batch_inputs, model, return_hiddens=True)
        #derivative of log(cross-entropy softmax)
        backward_inputs = outputs - batch_unflattened_labels

        gradient_weights = self.backward_pass(backward_inputs, hiddens, batch_inputs, model)
        
        if not check_gradient:
            if l2_regularization_const > 0.0:
                return gradient_weights / batch_size + model * l2_regularization_const
            return gradient_weights / batch_size
        
        ### below block checks gradient... only to be used if you think the gradient is incorrectly calculated ##############
        else:
            if l2_regularization_const > 0.0:
                gradient_weights += model * (l2_regularization_const * batch_size)
            sys.stdout.write("\r                                                                \r")
            print "checking gradient..."
            finite_difference_model = Recurrent_Neural_Network_Weight()
            finite_difference_model.init_zero_weights(self.model.get_architecture(), verbose=False)
            
            direction = Recurrent_Neural_Network_Weight()
            direction.init_zero_weights(self.model.get_architecture(), verbose=False)
            epsilon = 1E-5
            print "at initial hiddens"
            for index in range(direction.init_hiddens.size):
                direction.init_hiddens[0][index] = epsilon
                forward_loss = self.calculate_cross_entropy(self.forward_pass(batch_inputs, model = model + direction), batch_unflattened_labels)
                backward_loss = self.calculate_cross_entropy(self.forward_pass(batch_inputs, model = model - direction), batch_unflattened_labels)
                finite_difference_model.init_hiddens[0][index] = (forward_loss - backward_loss) / (2 * epsilon)
                direction.init_hiddens[0][index] = 0.0
            for key in direction.bias.keys():
                print "at bias key", key
                for index in range(direction.bias[key].size):
                    direction.bias[key][0][index] = epsilon
                    #print direction.norm()
                    forward_loss = self.calculate_cross_entropy(self.forward_pass(batch_inputs, model = model + direction), batch_unflattened_labels)
                    backward_loss = self.calculate_cross_entropy(self.forward_pass(batch_inputs, model = model - direction), batch_unflattened_labels)
                    finite_difference_model.bias[key][0][index] = (forward_loss - backward_loss) / (2 * epsilon)
                    direction.bias[key][0][index] = 0.0
            for key in direction.weights.keys():
                print "at weight key", key
                for index0 in range(direction.weights[key].shape[0]):
                    for index1 in range(direction.weights[key].shape[1]):
                        direction.weights[key][index0][index1] = epsilon
                        forward_loss = self.calculate_cross_entropy(self.forward_pass(batch_inputs, model = model + direction), batch_unflattened_labels)
                        backward_loss = self.calculate_cross_entropy(self.forward_pass(batch_inputs, model = model - direction), batch_unflattened_labels)
                        finite_difference_model.weights[key][index0][index1] = (forward_loss - backward_loss) / (2 * epsilon)
                        direction.weights[key][index0][index1] = 0.0
            
            print "calculated gradient for initial hiddens"
            print gradient_weights.init_hiddens
            print "finite difference approximation for initial hiddens"
            print finite_difference_model.init_hiddens
            
            print "calculated gradient for hidden bias"
            print gradient_weights.bias['hidden']
            print "finite difference approximation for hidden bias"
            print finite_difference_model.bias['hidden']
            
            print "calculated gradient for output bias"
            print gradient_weights.bias['output']
            print "finite difference approximation for output bias"
            print finite_difference_model.bias['output']
            
            print "calculated gradient for visible_hidden layer"
            print gradient_weights.weights['visible_hidden']
            print "finite difference approximation for visible_hidden layer"
            print finite_difference_model.weights['visible_hidden']
            
            print "calculated gradient for hidden_hidden layer"
            print gradient_weights.weights['hidden_hidden']
            print "finite difference approximation for hidden_hidden layer"
            print finite_difference_model.weights['hidden_hidden']
            
            print "calculated gradient for hidden_output layer"
            print gradient_weights.weights['hidden_output']
            print "finite difference approximation for hidden_output layer"
            print finite_difference_model.weights['hidden_output']
            
            sys.exit()
        ##########################################################

    def pearlmutter_forward_pass(self, inputs, unflattened_labels, direction, batch_size, hiddens=None, outputs=None, model=None, check_gradient=False, stop_at='output'): #need to test
        """let f be a function from inputs to outputs
        consider the weights to be a vector w of parameters to be optimized, (and direction d to be the same)
        pearlmutter_forward_pass calculates d' \jacobian_w f
        stop_at is either 'linear', 'output', or 'loss' """
        
        if model == None:
            model = self.model
        if hiddens == None or outputs == None:
            outputs, hiddens = self.forward_pass(inputs, model, return_hiddens=True)
            
        architecture = self.model.get_architecture()
        max_sequence_observations = inputs.shape[0]
        num_hiddens = architecture[1]
        num_sequences = inputs.shape[1]
        num_outs = architecture[2]
        hidden_deriv = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
        output_deriv = np.zeros((max_sequence_observations, num_sequences, num_outs))
        if stop_at == 'loss':
            loss_deriv = np.zeros(output_deriv.shape)
        
        #propagate hiddens
#        print model.init_hiddens.shape
        hidden_deriv[0,:,:] = (self.weight_matrix_multiply(inputs[0,:,:], direction.weights['visible_hidden'], direction.bias['hidden'])
                               + np.dot(model.init_hiddens, direction.weights['hidden_hidden']) 
                               + np.dot(direction.init_hiddens, model.weights['hidden_hidden'])) * (1 + hiddens[0,:,:]) * (1 - hiddens[0,:,:])
        linear_layer = (self.weight_matrix_multiply(hiddens[0,:,:], direction.weights['hidden_output'], 
                                                    direction.bias['output']) +
                        np.dot(hidden_deriv[0,:,:], model.weights['hidden_output']))
        if stop_at == 'linear':
            output_deriv[0,:,:] = linear_layer
        elif stop_at == 'output':
            output_deriv[0,:,:] = linear_layer * outputs[0,:,:] - outputs[0,:,:] * np.sum(linear_layer * outputs[0,:,:], axis=1)[:,np.newaxis]
#        if stop_at == 'loss':
#            output_deriv[model.num_layers+1] = -np.array([(hidden_deriv[model.num_layers][index, labels[index]] / hiddens[model.num_layers][index, labels[index]])[0] for index in range(batch_size)])
        for sequence_index in range(1, max_sequence_observations):
            sequence_input = inputs[sequence_index,:,:]
            hidden_deriv[sequence_index,:,:] = (self.weight_matrix_multiply(sequence_input, direction.weights['visible_hidden'], direction.bias['hidden'])
                                   + np.dot(hiddens[sequence_index-1,:,:], direction.weights['hidden_hidden']) 
                                   + np.dot(hidden_deriv[sequence_index-1,:,:], model.weights['hidden_hidden'])) * (1 + hiddens[sequence_index,:,:]) * (1 - hiddens[sequence_index,:,:])
            linear_layer = (self.weight_matrix_multiply(hiddens[sequence_index,:,:], direction.weights['hidden_output'], 
                                                        direction.bias['output']) +
                            np.dot(hidden_deriv[sequence_index,:,:], model.weights['hidden_output']))
            #find the observations where the sequence has ended, 
            #and then zero out hiddens and outputs, so nothing horrible happens during backprop, etc.
            zero_input = np.where(self.feature_sequence_lens <= sequence_index)
            hidden_deriv[sequence_index,zero_input,:] = 0.0
            output_deriv[sequence_index,zero_input,:] = 0.0
            if stop_at == 'linear':
                output_deriv[sequence_index,:,:] = linear_layer
            else:
                output_deriv[sequence_index,:,:] = linear_layer * outputs[sequence_index,:,:] - outputs[sequence_index,:,:] * np.sum(linear_layer * outputs[sequence_index,:,:], axis=1)[:,np.newaxis]
#            if stop_at == 'loss':
#                loss_deriv[sequence_index,:,:] = -np.array([(hidden_deriv[model.num_layers][index, labels[index]] / hiddens[model.num_layers][index, labels[index]])[0] for index in range(batch_size)])
        if not check_gradient:
            return output_deriv, hidden_deriv
        #compare with finite differences approximation
        else:
            epsilon = 1E-5
            if stop_at == 'linear':
                calculated = output_deriv
                finite_diff_forward = self.forward_pass(inputs, model = model + direction * epsilon, linear_output=True)
                finite_diff_backward = self.forward_pass(inputs, model = model - direction * epsilon, linear_output=True)
            elif stop_at == 'output':
                calculated = output_deriv
                finite_diff_forward = self.forward_pass(inputs, model = model + direction * epsilon)
                finite_diff_backward = self.forward_pass(inputs, model = model - direction * epsilon)
#            elif stop_at == 'loss':
#                calculated = hidden_deriv[model.num_layers + 1]
#                finite_diff_forward = -np.log([max(self.forward_pass(inputs, model = model + direction * epsilon).item((x,labels[x])),1E-12) for x in range(labels.size)]) 
#                finite_diff_backward =  -np.log([max(self.forward_pass(inputs, model = model - direction * epsilon).item((x,labels[x])),1E-12) for x in range(labels.size)]) 
            for seq in range(num_sequences):
                finite_diff_approximation = ((finite_diff_forward - finite_diff_backward) / (2 * epsilon))[:,seq,:]
                print "At sequence", seq
                print "pearlmutter calculation"
                print calculated[:,seq,:]
                print "finite differences approximation, epsilon", epsilon
                print finite_diff_approximation
            sys.exit()
    def calculate_per_example_cross_entropy(self, example_output, example_label):
        if example_label.size > 1:
            return -np.sum(np.log(np.clip(example_output, a_min=1E-12, a_max=1.0)) * example_label)
        else:
            return -np.log(np.clip(example_output[example_label], a_min=1E-12, a_max=1.0))
    def calculate_second_order_direction(self, inputs, unflattened_labels, batch_size, direction = None, model = None, second_order_type = None, 
                                         hiddens = None, outputs=None, check_direction = False, feature_sequence_lens = None, structural_damping_const = 0.0): #need to test
        #given an input direction direction, the function returns H*d, where H is the Hessian of the weight vector
        #the function does this efficient by using the Pearlmutter (1994) trick
        excluded_keys = {'bias': ['0'], 'weights': []}
        if model == None:
            model = self.model
        if direction == None:
            direction = self.calculate_gradient(inputs, unflattened_labels, False, model)
        if second_order_type == None:
            second_order_type='gauss-newton' #other option is 'hessian'
        if hiddens == None or outputs == None:
            outputs, hiddens = self.forward_pass(inputs, model, return_hiddens=True)   
        
        
        if second_order_type == 'gauss-newton':
            output_deriv, hidden_deriv = self.pearlmutter_forward_pass(inputs, unflattened_labels, direction, batch_size, hiddens, outputs, model, stop_at='output') #nbatch x nout
            second_order_direction = self.backward_pass(output_deriv, hiddens, inputs, model, structural_damping_const)
        elif second_order_type == 'hessian':
            output_deriv, hidden_deriv = self.pearlmutter_forward_pass(inputs, unflattened_labels, direction, batch_size, hiddens, outputs, model, stop_at='output') #nbatch x nout
            second_order_direction = self.pearlmutter_backward_pass(hidden_deriv, unflattened_labels, hiddens, model, direction)
        elif second_order_type == 'fisher':
            output_deriv, hidden_deriv = self.pearlmutter_forward_pass(inputs, unflattened_labels, direction, batch_size, hiddens, outputs, model, stop_at='loss')#nbatch x nout
            weight_vec = output_deriv - unflattened_labels
            weight_vec *= hidden_deriv[model.num_layers+1][:, np.newaxis] #TODO: fix this line
            second_order_direction = self.backward_pass(weight_vec, hiddens, inputs, model) 
        else:
            print second_order_type, "is not a valid type. Acceptable types are gauss-newton, hessian, and fisher... Exiting now..."
            sys.exit()
            
        if not check_direction:
            if self.l2_regularization_const > 0.0:
                return second_order_direction / batch_size + direction * self.l2_regularization_const
            return second_order_direction / batch_size
        
        ##### check direction only if you think there is a problem #######
        else:
            finite_difference_model = Recurrent_Neural_Network_Weight()
            finite_difference_model.init_zero_weights(self.model.get_architecture(), verbose=False)
            epsilon = 1E-5
            
            if second_order_type == 'gauss-newton':
                #assume that pearlmutter forward pass is correct because the function has a check_gradient flag to see if it's is
                sys.stdout.write("\r                                                                \r")
                sys.stdout.write("checking Gv\n"), sys.stdout.flush()
                linear_out = self.forward_pass(inputs, model = model, linear_output=True)
                num_examples = self.batch_size(feature_sequence_lens)
                finite_diff_forward = self.forward_pass(inputs, model = model + direction * epsilon, linear_output=True)
                finite_diff_backward = self.forward_pass(inputs, model = model - direction * epsilon, linear_output=True)
                finite_diff_jacobian_vec = (finite_diff_forward - finite_diff_backward) / (2 * epsilon)
                flat_finite_diff_jacobian_vec = self.flatten_output(finite_diff_jacobian_vec, feature_sequence_lens)
                flat_linear_out = self.flatten_output(linear_out, feature_sequence_lens)
                flat_labels = self.flatten_output(unflattened_labels, feature_sequence_lens)
                
                flat_finite_diff_HJv = np.zeros(flat_finite_diff_jacobian_vec.shape)
                
                num_outputs = flat_linear_out.shape[1]
                collapsed_hessian = np.zeros((num_outputs,num_outputs))
                for example_index in range(num_examples):
                    #calculate collapsed Hessian
                    direction1 = np.zeros(num_outputs)
                    direction2 = np.zeros(num_outputs)
                    for index1 in range(num_outputs):
                        for index2 in range(num_outputs):
                            direction1[index1] = epsilon
                            direction2[index2] = epsilon
                            example_label = np.array(flat_labels[example_index])
                            loss_plus_plus = self.calculate_per_example_cross_entropy(self.softmax(np.array([flat_linear_out[example_index] + direction1 + direction2])), example_label)
                            loss_plus_minus = self.calculate_per_example_cross_entropy(self.softmax(np.array([flat_linear_out[example_index] + direction1 - direction2])), example_label)
                            loss_minus_plus = self.calculate_per_example_cross_entropy(self.softmax(np.array([flat_linear_out[example_index] - direction1 + direction2])), example_label)
                            loss_minus_minus = self.calculate_per_example_cross_entropy(self.softmax(np.array([flat_linear_out[example_index] - direction1 - direction2])), example_label)
                            collapsed_hessian[index1,index2] = (loss_plus_plus + loss_minus_minus - loss_minus_plus - loss_plus_minus) / (4 * epsilon * epsilon)
                            direction1[index1] = 0.0
                            direction2[index2] = 0.0
#                    print collapsed_hessian
                    out = self.softmax(flat_linear_out[example_index:example_index+1])
#                    print np.diag(out[0]) - np.outer(out[0], out[0])
                    flat_finite_diff_HJv[example_index] += np.dot(collapsed_hessian, flat_finite_diff_jacobian_vec[example_index])
                    
                obs_so_far = 0
                for sequence_index, num_obs in enumerate(feature_sequence_lens):
                    print "at sequence index", sequence_index
                    #calculate J'd = J'HJv
                    update = Recurrent_Neural_Network_Weight()
                    update.init_zero_weights(self.model.get_architecture(), verbose=False)
                    for index in range(direction.init_hiddens.size):
                        update.init_hiddens[0][index] = epsilon
                        #print direction.norm()
                        forward_loss = self.forward_pass(inputs[:,sequence_index:sequence_index+1,:], model = model + update, linear_output=True)
                        backward_loss = self.forward_pass(inputs[:,sequence_index:sequence_index+1,:], model = model - update, linear_output=True)
                        for obs_index in range(num_obs):
                            example_index = obs_so_far + obs_index
                            finite_difference_model.init_hiddens[0][index] += np.dot((forward_loss[obs_index,0,:] - backward_loss[obs_index,0,:]) / (2 * epsilon), 
                                                                                     flat_finite_diff_HJv[example_index])
                            update.init_hiddens[0][index] = 0.0
                    for key in direction.bias.keys():
                        print "at bias key", key
                        for index in range(direction.bias[key].size):
                            update.bias[key][0][index] = epsilon
                            #print direction.norm()
                            forward_loss = self.forward_pass(inputs[:,sequence_index:sequence_index+1,:], model = model + update, linear_output=True)
                            backward_loss = self.forward_pass(inputs[:,sequence_index:sequence_index+1,:], model = model - update, linear_output=True)
                            for obs_index in range(num_obs):
                                example_index = obs_so_far + obs_index
                                finite_difference_model.bias[key][0][index] += np.dot((forward_loss[obs_index,0,:] - backward_loss[obs_index,0,:]) / (2 * epsilon), 
                                                                                      flat_finite_diff_HJv[example_index])
                            update.bias[key][0][index] = 0.0
                    for key in direction.weights.keys():
                        print "at weight key", key
                        for index0 in range(direction.weights[key].shape[0]):
                            for index1 in range(direction.weights[key].shape[1]):
                                update.weights[key][index0][index1] = epsilon
                                forward_loss = self.forward_pass(inputs[:,sequence_index:sequence_index+1,:], model= model + update, linear_output=True)
                                backward_loss = self.forward_pass(inputs[:,sequence_index:sequence_index+1,:], model= model - update, linear_output=True)
                                for obs_index in range(num_obs):
                                    example_index = obs_so_far + obs_index
                                    finite_difference_model.weights[key][index0][index1] += np.dot((forward_loss[obs_index,0,:] - backward_loss[obs_index,0,:]) / (2 * epsilon), 
                                                                                                   flat_finite_diff_HJv[example_index])
                                update.weights[key][index0][index1] = 0.0
                    obs_so_far += num_obs
            elif second_order_type == 'hessian':
                sys.stdout.write("\r                                                                \r")
                sys.stdout.write("checking Hv\n"), sys.stdout.flush()
                for batch_index in range(batch_size):
                    #assume that gradient calculation is correct
                    print "at batch index", batch_index
                    update = Recurrent_Neural_Network_Weight()
                    update.init_zero_weights(self.model.get_architecture(), verbose=False)
                    
                    current_gradient = self.calculate_gradient(inputs[:,batch_index:batch_index+1,:], unflattened_labels[:,batch_index:batch_index+1,:], batch_size, model=model, l2_regularization_const = 0.)
                    
                    for key in finite_difference_model.bias.keys():
                        for index in range(direction.bias[key].size):
                            update.bias[key][0][index] = epsilon
                            forward_loss = self.calculate_gradient(inputs[:,batch_index:batch_index+1,:], unflattened_labels[:,batch_index:batch_index+1,:], batch_size, 
                                                                   model = model + update, l2_regularization_const = 0.)
                            backward_loss = self.calculate_gradient(inputs[:,batch_index:batch_index+1,:], unflattened_labels[:,batch_index:batch_index+1,:], batch_size, 
                                                                    model = model - update, l2_regularization_const = 0.)
                            finite_difference_model.bias[key][0][index] += direction.dot((forward_loss - backward_loss) / (2 * epsilon), excluded_keys)
                            update.bias[key][0][index] = 0.0
        
                    for key in finite_difference_model.weights.keys():
                        for index0 in range(direction.weights[key].shape[0]):
                            for index1 in range(direction.weights[key].shape[1]):
                                update.weights[key][index0][index1] = epsilon
                                forward_loss = self.calculate_gradient(inputs[:,batch_index:batch_index+1,:], unflattened_labels[:,batch_index:batch_index+1,:], batch_size, 
                                                                       model = model + update, l2_regularization_const = 0.) 
                                backward_loss = self.calculate_gradient(inputs[:,batch_index:batch_index+1,:], unflattened_labels[:,batch_index:batch_index+1,:], batch_size, 
                                                                        model = model - update, l2_regularization_const = 0.)
                                finite_difference_model.weights[key][index0][index1] += direction.dot((forward_loss - backward_loss) / (2 * epsilon), excluded_keys)
                                update.weights[key][index0][index1] = 0.0
            elif second_order_type == 'fisher':
                sys.stdout.write("\r                                                                \r")
                sys.stdout.write("checking Fv\n"), sys.stdout.flush()
                for batch_index in range(batch_size):
                    #assume that gradient calculation is correct
                    print "at batch index", batch_index
                    current_gradient = self.calculate_gradient(inputs[:,batch_index:batch_index+1,:], unflattened_labels[:,batch_index:batch_index+1,:], batch_size, model = model, l2_regularization_const = 0.)                
                    finite_difference_model += current_gradient * current_gradient.dot(direction, excluded_keys)
            
            print "calculated second order direction for init hiddens"
            print second_order_direction.init_hiddens
            print "finite difference approximation for init hiddens"
            print finite_difference_model.init_hiddens
            
            for bias_cur_layer in direction.bias.keys():
                print "calculated second order direction for bias", bias_cur_layer
                print second_order_direction.bias[bias_cur_layer]
                print "finite difference approximation for bias", bias_cur_layer
                print finite_difference_model.bias[bias_cur_layer]
            for weight_cur_layer in finite_difference_model.weights.keys():
                print "calculated second order direction for weights", weight_cur_layer
                print second_order_direction.weights[weight_cur_layer]
                print "finite difference approximation for weights", weight_cur_layer
                print finite_difference_model.weights[weight_cur_layer]
            sys.exit()
        ##########################################################