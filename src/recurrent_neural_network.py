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
        return [self.bias[str(layer_num)].size for layer_num in range(self.num_layers+1) ]
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
        
        self.init_hiddens = initial_bias_min + initial_bias_range * np.random.random_sample((1,architecture[1]))
        
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
            nn_output.init_hiddens = self.init_hiddens + addend.init_weights
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
            nn_output.init_hiddens = self.init_hiddens - subtrahend.init_weights
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
        return Recurrent_Neural_Network_Weight(self.num_layers, self.weights, self.bias, self.weight_type)
    def __deepcopy__(self, memo):
        return Recurrent_Neural_Network_Weight(copy.deepcopy(self.num_layers, memo), copy.deepcopy(self.weights,memo), 
                                     copy.deepcopy(self.bias,memo), copy.deepcopy(self.weight_type,memo))
        


class Recurrent_Neural_Network(object, Vector_Math):
    """features are stored in format max_seq_len x nvis x nseq where n_max_obs is the maximum number of observations per sequence
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
        self.all_variables['train'] = self.required_variables['train'] + ['label_file_name', 'hiddens_structure', 'weight_matrix_name', 
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
            sequence_len = feature_data['sequence_lengths']
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
            return  label_data[:,2], label_data[:,1]#in MATLAB format
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
        if len(self.labels.shape) != 1 and ((len(self.labels.shape) == 2 and self.labels.shape[1] != 1) or len(self.labels.shape) > 2):
            print "labels need to be in (n_samples) or (n_samples,1) format and the shape of labels is ", self.labels.shape, "... Exiting now"
            sys.exit()
        if self.labels.size != self.features.shape[0] * self.features.shape[1]:
            print "Number of examples in feature file: ", self.features.shape[0] * self.features.shape[1], " does not equal size of label file, ", self.labels.size, "... Exiting now"
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
    def forward_pass_linear(self, inputs, verbose=True, model=None):
        #to test finite differences calculation for pearlmutter forward pass, just like forward pass, except it spits linear outputs
        if model == None:
            model = self.model
        architecture = self.model.get_architecture()
        max_sequence_observations = inputs.shape[0]
        num_hiddens = architecture[1]
        num_sequences = inputs.shape[2]
        num_outs = architecture[2]
        hiddens = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
        outputs = np.zeros((max_sequence_observations, num_sequences, num_outs))
        
        #propagate hiddens
        hiddens[0,:,:] = self.forward_layer(inputs[0,:,:], self.model.weights['visible_hidden'], self.model.bias['hidden'], 
                                            self.model.weight_type['visible_hidden'], self.model.init_hiddens, 
                                            self.model.weights['hidden_hidden'])
        outputs[0,:,:] = self.forward_layer(hiddens[0,:,:], self.model.weights['hidden_output'], self.model.bias['output'], 
                                            self.model.weight_type['hidden_output'])
        for sequence_index in range(1, max_sequence_observations):
            sequence_input = inputs[sequence_index,:,:]
            hiddens[sequence_index,:,:] = self.forward_layer(sequence_input, self.model.weights['visible_hidden'], self.model.bias['hidden'], 
                                                             self.model.weight_type['visible_hidden'], hiddens[sequence_index-1,:,:], 
                                                             self.model.weights['hidden_hidden'])
            outputs[sequence_index,:,:] = self.forward_layer(hiddens[sequence_index,:,:], self.model.weights['hidden_output'], self.model.bias['output'], 
                                                             'linear')
            #find the observations where the sequence has ended, 
            #and then zero out hiddens and outputs, so nothing horrible happens during backprop, etc.
            zero_input = np.where(self.feature_sequence_lens >= sequence_index)
            hiddens[sequence_index,:,zero_input] = 0.0
            outputs[sequence_index,:,zero_input] = 0.0

        del hiddens
        return outputs
    def forward_pass(self, inputs, model=None, return_hiddens=False): #completed
        """forward pass each layer starting with feature level
        inputs in the form n_max_obs x n_seq x n_vis"""
        if model == None:
            model = self.model
        architecture = self.model.get_architecture()
        max_sequence_observations = inputs.shape[0]
        num_hiddens = architecture[1]
        num_sequences = inputs.shape[2]
        num_outs = architecture[2]
        hiddens = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
        outputs = np.zeros((max_sequence_observations, num_sequences, num_outs))
        
        #propagate hiddens
        hiddens[0,:,:] = self.forward_layer(inputs[0,:,:], self.model.weights['visible_hidden'], self.model.bias['hidden'], 
                                            self.model.weight_type['visible_hidden'], self.model.init_hiddens, 
                                            self.model.weights['hidden_hidden'])
        outputs[0,:,:] = self.forward_layer(hiddens[0,:,:], self.model.weights['hidden_output'], self.model.bias['output'], 
                                            self.model.weight_type['hidden_output'])
        for sequence_index in range(1, max_sequence_observations):
            sequence_input = inputs[sequence_index,:,:]
            hiddens[sequence_index,:,:] = self.forward_layer(sequence_input, self.model.weights['visible_hidden'], self.model.bias['hidden'], 
                                                             self.model.weight_type['visible_hidden'], hiddens[sequence_index-1,:,:], 
                                                             self.model.weights['hidden_hidden'])
            outputs[sequence_index,:,:] = self.forward_layer(hiddens[sequence_index,:,:], self.model.weights['hidden_output'], self.model.bias['output'], 
                                                             self.model.weight_type['hidden_output'])
            #find the observations where the sequence has ended, 
            #and then zero out hiddens and outputs, so nothing horrible happens during backprop, etc.
            zero_input = np.where(self.feature_sequence_lens >= sequence_index)
            hiddens[sequence_index,:,zero_input] = 0.0
            outputs[sequence_index,:,zero_input] = 0.0
        if return_hiddens:
            return outputs, hiddens
        else:
            del hiddens
            return outputs
    def flatten_output(self, output):
        """outputs in the form of max_obs_seq x n_outs x n_seq get converted to form
        n_data x n_outs, so we can calculate classification accuracy and cross-entropy
        """
        num_outs = output.shape[1]
        num_seq = output.shape[2]
        flat_output = np.zeros((np.sum(self.feature_sequence_lens), num_outs))
        cur_index = 0
        for seq_index in range(num_seq):
            flat_output[cur_index:cur_index+self.feature_sequence_lens[seq_index], :] = copy.deepcopy(output[:self.feature_sequence_lens[seq_index], :, seq_index])
            cur_index += self.feature_sequence_lens[seq_index]
        
        return flat_output
    def calculate_cross_entropy(self, flat_output, labels): #completed, expensive, should be compiled
        return -np.sum(np.log([max(flat_output.item((x,labels[x])),1E-12) for x in range(labels.size)]))
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
        self.num_training_examples = self.features.shape[0]
        self.check_keys(config_dictionary)
        #read label file
        self.label_file_name = self.default_variable_define(config_dictionary, 'label_file_name', arg_type='string', error_string="No label_file_name defined, can only do pretraining",exit_if_no_default=False)
        if self.label_file_name != None:
            self.labels, self.labels_sent_id = self.read_label_file()
            self.check_labels()
            self.unflattened_labels = self.unflatten_labels(self.labels, self.sentence_ids) 
        else:
            del self.label_file_name        

        #initialize weights
        self.weight_matrix_name = self.default_variable_define(config_dictionary, 'weight_matrix_name', exit_if_no_default=False)

        if self.weight_matrix_name != None:
            print "Since weight_matrix_name is defined, ignoring possible value of hiddens_structure"
            self.model.open_weights(self.weight_matrix_name)
        else: #initialize model
            del self.weight_matrix_name
            
            self.hiddens_structure = self.default_variable_define(config_dictionary, 'hiddens_structure', arg_type='int_comma_string', exit_if_no_default=True)
            architecture = [self.features.shape[1]] + self.hiddens_structure
            if hasattr(self, 'labels'):
                architecture.append(np.max(self.labels)+1) #will have to change later if I have soft weights
                
            self.initial_weight_max = self.default_variable_define(config_dictionary, 'initial_weight_max', arg_type='float', default_value=0.1)
            self.initial_weight_min = self.default_variable_define(config_dictionary, 'initial_weight_min', arg_type='float', default_value=-0.1)
            self.initial_bias_max = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=-2.2)
            self.initial_bias_min = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=-2.4)
            self.model.init_random_weights(architecture, self.initial_bias_max, self.initial_bias_min, 
                                           self.initial_weight_min, self.initial_weight_max, last_layer_logistic=hasattr(self,'labels'))
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
                                                                acceptable_values=['steepest_descent', 'conjugate_gradient', 'krylov_subspace', 'truncated_newton'])
            self.backprop_batch_size = self.default_variable_define(config_dictionary, 'backprop_batch_size', default_value=2048, arg_type='int')
            self.l2_regularization_const = self.default_variable_define(config_dictionary, 'l2_regularization_const', arg_type='float', default_value=0.0, exit_if_no_default=False)
            
            if self.backprop_method == 'steepest_descent':
                self.steepest_learning_rate = self.default_variable_define(config_dictionary, 'steepest_learning_rate', default_value=[0.008, 0.004, 0.002, 0.001], arg_type='float_comma_string')
            else:
                self.num_epochs = self.default_variable_define(config_dictionary, 'num_epochs', default_value=20, arg_type='int')
                if self.backprop_method == 'conjugate_gradient':
                    self.num_line_searches = self.default_variable_define(config_dictionary, 'num_line_searches', default_value=20, arg_type='int')
                    self.conjugate_max_iterations = self.default_variable_define(config_dictionary, 'conjugate_max_iterations', default_value=3, 
                                                                                 arg_type='int')
                    self.conjugate_const_type = self.default_variable_define(config_dictionary, 'conjugate_const_type', arg_type='string', default_value='polak-ribiere', 
                                                                             acceptable_values = ['polak-ribiere', 'polak-ribiere+', 'hestenes-stiefel', 'fletcher-reeves'])
                    self.armijo_const = self.default_variable_define(config_dictionary, 'armijo_const', arg_type='float', default_value=0.1)
                    self.wolfe_const = self.default_variable_define(config_dictionary, 'wolfe_const', arg_type='float', default_value=0.2)
                elif self.backprop_method == 'krylov_subspace':
                    self.num_line_searches = self.default_variable_define(config_dictionary, 'num_line_searches', default_value=20, arg_type='int')
                    self.second_order_matrix = self.default_variable_define(config_dictionary, 'second_order_matrix', arg_type='string', default_value='gauss-newton', 
                                                                            acceptable_values=['gauss-newton', 'hessian', 'fisher'])
                    self.krylov_num_directions = self.default_variable_define(config_dictionary, 'krylov_num_directions', arg_type='int', default_value=20, 
                                                                              acceptable_values=range(2,2000))
                    self.krylov_num_batch_splits = self.default_variable_define(config_dictionary, 'krylov_num_batch_splits', arg_type='int', default_value=self.krylov_num_directions, 
                                                                                acceptable_values=range(2,2000))
                    self.krylov_num_bfgs_epochs = self.default_variable_define(config_dictionary, 'krylov_num_bfgs_epochs', arg_type='int', default_value=self.krylov_num_directions)
                    self.krylov_use_hessian_preconditioner = self.default_variable_define(config_dictionary, 'krylov_use_hessian_preconditioner', arg_type='boolean', default_value=True)
                    if self.krylov_use_hessian_preconditioner:
                        self.krylov_eigenvalue_floor_const = self.default_variable_define(config_dictionary, 'krylov_eigenvalue_floor_const', arg_type='float', default_value=1E-4)
                    self.use_fisher_preconditioner = self.default_variable_define(config_dictionary, 'use_fisher_preconditioner', arg_type='boolean', default_value=False)
                    if self.use_fisher_preconditioner:
                        self.fisher_preconditioner_floor_val = self.default_variable_define(config_dictionary, 'fisher_preconditioner_floor_val', arg_type='float', default_value=1E-4)
                    self.armijo_const = self.default_variable_define(config_dictionary, 'armijo_const', arg_type='float', default_value=0.0001)
                    self.wolfe_const = self.default_variable_define(config_dictionary, 'wolfe_const', arg_type='float', default_value=0.9)
                elif self.backprop_method == 'truncated_newton':
                    self.second_order_matrix = self.default_variable_define(config_dictionary, 'second_order_matrix', arg_type='string', default_value='gauss-newton', 
                                                                            acceptable_values=['gauss-newton', 'hessian'])
                    self.use_fisher_preconditioner = self.default_variable_define(config_dictionary, 'use_fisher_preconditioner', arg_type='boolean', default_value=False)
                    if self.use_fisher_preconditioner:
                        self.fisher_preconditioner_floor_val = self.default_variable_define(config_dictionary, 'fisher_preconditioner_floor_val', arg_type='float', default_value=1E-4)
                    self.truncated_newton_num_cg_epochs = self.default_variable_define(config_dictionary, 'truncated_newton_num_cg_epochs', arg_type='int', default_value=20)
                    self.truncated_newton_init_damping_factor = self.default_variable_define(config_dictionary, 'truncated_newton_init_damping_factor', arg_type='float', default_value=0.1)
        self.dump_config_vals()
    def unflatten_labels(self, labels, sentence_ids):
        num_frames_per_sentence = np.bincount(np.ravel(sentence_ids))
        num_outs = np.unique(labels)
        max_num_frames_per_sentence = np.max(num_frames_per_sentence)
        unflattened_labels = np.zeros((max_num_frames_per_sentence, num_outs, np.max(sentence_ids) + 1)) #add one because first sentence starts at 0
        current_sentence_id = 0
        observation_index = 0
        for label, sentence_id in zip(labels,sentence_ids):
            if sentence_id != current_sentence_id:
                current_sentence_id = sentence_id
                observation_index = 0
            unflattened_labels[observation_index, label, sentence_id] = 1.0
            observation_index += 1
        return unflattened_labels
    def backward_pass(self, backward_inputs, hiddens, visibles, model=None): #need to test
        
        if model == None:
            model = self.model
        output_model = Recurrent_Neural_Network_Weight(num_layers=model.num_layers)
        output_model.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
        
        n_obs = backward_inputs.shape[0]
        n_outs = backward_inputs.shape[1]
        n_hids = hiddens.shape[1]
        n_vis = visibles.shape[1]
        n_seq = backward_inputs.shape[2]
        #backward_inputs - n_obs x n_seq x n_outs
        #hiddens - n_obs  x n_seq x n_hids
        flat_outputs = np.reshape(np.transpose(backward_inputs, axes=(1,0,2)),(n_obs * n_seq,n_outs))
        flat_hids = np.reshape(np.transpose(hiddens, axes=(1,0,2)),(n_obs * n_seq,n_hids))
        flat_vis = np.reshape(np.transpose(visibles, axes=(1,0,2)),(n_obs * n_seq,n_vis))
        #average layers in batch
        
        output_model.bias['output'][0] = np.sum(flat_outputs, axis=0)
        output_model.weights['hidden_output'] = np.dot(flat_hids.T, flat_outputs)
        
        #HERE is where the fun begins... will need to store gradient updates in the form of dL/da_i 
        #(where a is the pre-nonlinear layer for updates to the hidden hidden and input hidden weight matrices
        
        pre_nonlinearity_hiddens = np.dot(backward_inputs, model.weights['hidden_output'].T) * (1 - np.tan(hiddens) ** 2)
        
        for observation_index in range(n_obs-1)[::-1]:
            pre_nonlinearity_hiddens[observation_index,:,:] += (np.dot(pre_nonlinearity_hiddens[observation_index+1,:,:],
                                                                       model.weights['hidden_hidden'].T) * 
                                                                (1 - np.tan(hiddens[observation_index+1,:,:]) ** 2))
        
        flat_pre_nonlinearity_hiddens = np.reshape(np.transpose(pre_nonlinearity_hiddens, axes=(1,0,2)),(n_obs * n_seq,n_hids))
        
        output_model.bias['hidden'][0] = np.sum(flat_pre_nonlinearity_hiddens, axis=0)
        output_model.weights['hidden_hidden'] = np.dot(flat_pre_nonlinearity_hiddens, flat_hids)
        
        output_model.weights['visible_hidden'] = np.dot(flat_pre_nonlinearity_hiddens, flat_vis)
        output_model.init_hiddens = np.sum(np.dot(pre_nonlinearity_hiddens[observation_index+1,:,:], model.weights['hidden_hidden'].T), axis=0)
        
        return output_model
        
    def calculate_gradient(self, batch_inputs, batch_unflattened_labels, batch_size, check_gradient=False, 
                           model=None, l2_regularization_const = 0.0): 
        #need to check regularization
        #calculate gradient with particular Neural Network model. If None is specified, will use current weights (i.e., self.model)
        excluded_keys = {'bias':['0'], 'weights':[]} #will have to change this later
        if model == None:
            model = self.model
        
        outputs, hiddens = self.forward_pass(batch_inputs, model, return_hiddens=True)
        
        #derivative of log(cross-entropy softmax)
        backward_inputs = outputs - batch_unflattened_labels

        gradient_weights = self.backward_pass(backward_inputs, hiddens, batch_inputs, model)
        
        if not check_gradient:
            if l2_regularization_const > 0.0:
                return gradient_weights / batch_size + model * l2_regularization_const
            return gradient_weights / batch_size

    def pearlmutter_forward_pass(self, labels, inputs, hiddens, outputs, model, direction, batch_size, check_gradient=False, stop_at='output'): #need to test
        """let f be a function from inputs to outputs
        consider the weights to be a vector w of parameters to be optimized, (and direction d to be the same)
        pearlmutter_forward_pass calculates d' \jacobian_w f
        stop_at is either 'linear', 'output', or 'loss' """
        
        architecture = self.model.get_architecture()
        max_sequence_observations = inputs.shape[0]
        num_hiddens = architecture[1]
        num_sequences = inputs.shape[2]
        num_outs = architecture[2]
        hidden_deriv = np.zeros((max_sequence_observations, num_sequences, num_hiddens))
        output_deriv = np.zeros((max_sequence_observations, num_sequences, num_outs))
        if stop_at == 'loss':
            loss_deriv = np.zeros(output_deriv.shape)
        
        #propagate hiddens
        hidden_deriv[0,:,:] = (self.weight_matrix_multiply(inputs[0,:,:], direction.weights['visible_hidden'], direction.bias['hidden'])
                               + np.dot(direction.weights['hidden_hidden'], model.init_hiddens) 
                               + np.dot(model.weights['hidden_hidden'], direction.init_hiddens))
        linear_layer = (self.weight_matrix_multiply(hiddens[0,:,:], direction.weights['hidden_output'], 
                                                    direction.bias['output']) +
                        np.dot(hidden_deriv[0,:,:], model.weights['hidden_output']))
        if stop_at == 'linear':
            output_deriv[0,:,:] = linear_layer
        elif stop_at == 'output':
            output_deriv[0,:,:] = linear_layer * outputs[0,:,:] - outputs[0,:,:] * np.sum(linear_layer * outputs[0,:,:], axis=1)[:,np.newaxis]
#        if stop_at == 'loss':
#            output_deriv[model.num_layers+1] = -np.array([(hidden_deriv[model.num_layers][index, labels[index]] / hiddens[model.num_layers][index, labels[index]])[0] for index in range(batch_size)])
#        hiddens[0,:,:] = self.forward_layer(inputs[0,:,:], self.model.weights['visible_hidden'], self.model.bias['hidden'], 
#                                            self.model.weight_type['visible_hidden'], self.model.init_hiddens, 
#                                            self.model.weights['hidden_hidden'])
#        
#        outputs[0,:,:] = self.forward_layer(hiddens[0,:,:], self.model.weights['hidden_output'], self.model.bias['output'], 
#                                            self.model.weight_type['hidden_output'])
        for sequence_index in range(1, max_sequence_observations):
            sequence_input = inputs[sequence_index,:,:]
            hidden_deriv[sequence_index,:,:] = (self.weight_matrix_multiply(sequence_input, direction.weights['visible_hidden'], direction.bias['hidden'])
                                   + np.dot(direction.weights['hidden_hidden'], hiddens[sequence_index-1,:,:]) 
                                   + np.dot(model.weights['hidden_hidden'], hidden_deriv[sequence_index-1,:,:]))
            linear_layer = (self.weight_matrix_multiply(hiddens[sequence_index,:,:], direction.weights['hidden_output'], 
                                                        direction.bias['output']) +
                            np.dot(hidden_deriv[sequence_index,:,:], model.weights['hidden_output']))
#            hiddens[sequence_index,:,:] = self.forward_layer(sequence_input, self.model.weights['visible_hidden'], self.model.bias['hidden'], 
#                                                             self.model.weight_type['visible_hidden'], hiddens[sequence_index-1,:,:], 
#                                                             self.model.weights['hidden_hidden'])
#            outputs[sequence_index,:,:] = self.forward_layer(hiddens[sequence_index,:,:], self.model.weights['hidden_output'], self.model.bias['output'], 
#                                                             self.model.weight_type['hidden_output'])
            #find the observations where the sequence has ended, 
            #and then zero out hiddens and outputs, so nothing horrible happens during backprop, etc.
            zero_input = np.where(self.feature_sequence_lens >= sequence_index)
            hidden_deriv[sequence_index,:,zero_input] = 0.0
            output_deriv[sequence_index,:,zero_input] = 0.0
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
            epsilon = 1E-10
            if stop_at == 'linear':
                calculated = hidden_deriv[model.num_layers]
                finite_diff_forward = self.forward_pass_linear(inputs, verbose=False, model = model + direction * epsilon)
                finite_diff_backward = self.forward_pass_linear(inputs, verbose=False, model = model - direction * epsilon)
            elif stop_at == 'output':
                calculated = hidden_deriv[model.num_layers]
                finite_diff_forward = self.forward_pass(inputs, verbose=False, model = model + direction * epsilon)
                finite_diff_backward = self.forward_pass(inputs, verbose=False, model = model - direction * epsilon)
            elif stop_at == 'loss':
                calculated = hidden_deriv[model.num_layers + 1]
                finite_diff_forward = -np.log([max(self.forward_pass(inputs, verbose=False, model = model + direction * epsilon).item((x,labels[x])),1E-12) for x in range(labels.size)]) 
                finite_diff_backward =  -np.log([max(self.forward_pass(inputs, verbose=False, model = model - direction * epsilon).item((x,labels[x])),1E-12) for x in range(labels.size)]) 
            print "pearlmutter calculation"
            print calculated
            print "finite differences approximation, epsilon", epsilon
            print ((finite_diff_forward - finite_diff_backward) / (2 * epsilon))
            sys.exit()
        ######################################################################
        
        ### below block checks gradient... only to be used if you think the gradient is incorrectly calculated ##############
#        else:
#            if l2_regularization_const > 0.0:
#                gradient_weights += model * (l2_regularization_const * batch_size)
#            sys.stdout.write("\r                                                                \r")
#            print "checking gradient..."
#            finite_difference_model = Recurrent_Neural_Network_Weight(num_layers=model.num_layers)
#            finite_difference_model.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
#            
#            direction = Recurrent_Neural_Network_Weight(num_layers=model.num_layers)
#            direction.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
#            epsilon = 1E-5
#            for key in direction.bias.keys():
#                print "at bias key", key
#                for index in range(direction.bias[key].size):
#                    direction.bias[key][0][index] = epsilon
#                    #print direction.norm()
#                    forward_loss = self.calculate_loss(batch_inputs, batch_labels, model = model + direction) #self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False, model = model + direction), batch_labels)
#                    backward_loss = self.calculate_loss(batch_inputs, batch_labels, model = model - direction) #self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False, model = model - direction), batch_labels)
#                    finite_difference_model.bias[key][0][index] = (forward_loss - backward_loss) / (2 * epsilon)
#                    direction.bias[key][0][index] = 0.0
#            for key in direction.weights.keys():
#                print "at weight key", key
#                for index0 in range(direction.weights[key].shape[0]):
#                    for index1 in range(direction.weights[key].shape[1]):
#                        direction.weights[key][index0][index1] = epsilon
#                        forward_loss = self.calculate_loss(batch_inputs, batch_labels, model = model + direction) #self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False, model = model + direction), batch_labels)
#                        backward_loss = self.calculate_loss(batch_inputs, batch_labels, model = model - direction) #self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False, model = model - direction), batch_labels)
#                        finite_difference_model.weights[key][index0][index1] = (forward_loss - backward_loss) / (2 * epsilon)
#                        direction.weights[key][index0][index1] = 0.0
#            
#            for layer_num in range(model.num_layers,0,-1):
#                weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
#                bias_cur_layer = str(layer_num)
#                print "calculated gradient for bias", bias_cur_layer
#                print gradient_weights.bias[bias_cur_layer]
#                print "finite difference approximation for bias", bias_cur_layer
#                print finite_difference_model.bias[bias_cur_layer]
#                print "calculated gradient for weights", weight_cur_layer
#                print gradient_weights.weights[weight_cur_layer]
#                print "finite difference approximation for weights", weight_cur_layer
#                print finite_difference_model.weights[weight_cur_layer]
#            print "calculated gradient for bias 0"
#            print gradient_weights.bias['0']
#            print "finite difference approximation for bias 0"
#            print finite_difference_model.bias['0']
#            sys.exit()
        ########################################################## 
                        
#    def forward_recurrent_layer(self, layer_inputs, layer_weights, layer_biases, start_sequence_flags,
#                                hidden_inits, hidden_weights, layer_weight_type):
#        outputs = self.weight_matrix_multiply(layer_inputs, layer_weights, layer_biases)
#        if layer_weight_type == 'logistic':
#            for feature_index, start_flag in enumerate(start_sequence_flags):
#                if start_flag == 1:
#                    outputs[feature_index] = self.softmax(outputs[feature_index] + np.dot(hidden_inits, hidden_weights))
#                else:
#                    outputs[feature_index] = self.softmax(outputs[feature_index] + np.dot(outputs[feature_index-1], hidden_weights))
#        elif layer_weight_type == 'rbm_gaussian_bernoulli' or layer_weight_type == 'rbm_bernoulli_bernoulli':
#            for feature_index, start_flag in enumerate(start_sequence_flags):
#                if start_flag == 1:
#                    outputs[feature_index] = self.sigmoid(outputs[feature_index] + np.dot(hidden_inits, hidden_weights))
#                else:
#                    outputs[feature_index] = self.sigmoid(outputs[feature_index] + np.dot(outputs[feature_index-1], hidden_weights))
#        #MAY NOT BE USEFUL,added to test finite differences calculation for pearlmutter forward pass
#        elif layer_weight_type == 'linear':
#            for feature_index, start_flag in enumerate(start_sequence_flags):
#                if start_flag == 1:
#                    outputs[feature_index] += np.dot(hidden_inits, hidden_weights)
#                else:
#                    outputs[feature_index] += np.dot(outputs[feature_index-1], hidden_weights)
#        else:
#            print "weight_type", layer_weight_type, "is not a valid layer type.",
#            print "Valid layer types are", self.model.valid_layer_types,"Exiting now..."
#            sys.exit()
#        return outputs
#    
#        def recurrent_forward_pass(self, inputs, verbose=True, model=None):
#        """forward pass for a recurrent neural network"""
#        if model == None:
#            model = self.model 
#        cur_layer = inputs
#        for layer_num in range(1,model.num_layers+1):
#            if verbose:
#                print "At layer", layer_num, "of", model.num_layers
#            weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
#            weight_cur_layer_recurrent = ''.join([str(layer_num),str(layer_num),'_recurrent'])
#            cur_layer_recurrent_inits = ''.join([str(layer_num),'_recurrent'])
#            bias_cur_layer = str(layer_num)
#            if weight_cur_layer_recurrent in model.weights.keys():
#                cur_layer = self.forward_recurrent_layer(self, cur_layer, model.weights[weight_cur_layer], model.bias[bias_cur_layer], self.start_sequence_flags,
#                                                    model.bias[cur_layer_recurrent_inits], model.weights[weight_cur_layer_recurrent], model.weight_type[weight_cur_layer])
#            else:
#                cur_layer = self.forward_layer(cur_layer, model.weights[weight_cur_layer], 
#                                               model.bias[bias_cur_layer], model.weight_type[weight_cur_layer])
#        return cur_layer