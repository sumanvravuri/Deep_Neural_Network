'''

Created on Jul 20, 2012

@author: sumanravuri

Deep Neural Network with arbitrary number of hidden layers with these features:
* pretraining
* backprop with either stochastic or conjugate gradient descent, or 2nd order methods
* L2 regularization
* Early Stopping
* CUDA speedup

'''

import sys
import numpy as np
import scipy.io as sp
import scipy.linalg as sl
import scipy.optimize as sopt
import math
import copy
import argparse

class Vector_Math:
    #math functions
    def sigmoid(self,inputs): #completed, expensive, should be compiled
        return 1./(1+np.exp(-inputs)) #1/(1+e^-X)
    def softmax(self, inputs): #completed, expensive, should be compiled
        #subtracting max value of each data point below for numerical stability
        #exp_inputs = np.exp(inputs - np.transpose(np.tile(np.max(inputs,axis=1), (inputs.shape[1],1))))
        exp_inputs = np.exp(inputs - np.max(inputs,axis=1)[:,np.newaxis])
        return exp_inputs / np.sum(exp_inputs, axis=1)[:, np.newaxis]
    def weight_matrix_multiply(self,inputs,weights,biases): #completed, expensive, should be compiled
        #print "input dims are ", inputs.shape
        #print "weight dims are ", weights.shape
        #print "bias dims are ", biases.shape
        #return np.dot(inputs,weights)+np.tile(biases, (inputs.shape[0],1))
        return np.dot(inputs,weights) + biases#[np.newaxis, :]
class Neural_Network_Weight(object):
    def __init__(self, num_layers=0, weights=None, bias=None, weight_type=None):
        #num_layers
        #weights - actual Neural Network weights, a dictionary with keys corresponding to layer, ie. weights['01'], weights['12'], etc. each numpy array
        #bias - NN biases, again a dictionary stored as bias['0'], bias['1'], bias['2'], etc.
        #weight_type - optional command indexed by same keys weights, possible optionals are 'rbm_gaussian_bernoullli', 'rbm_bernoulli_bernoulli', 'logistic', 'convolutional', or 'pooling'
        self.valid_layer_types = dict()
        self.valid_layer_types['all'] = ['rbm_gaussian_bernoulli', 'rbm_bernoulli_bernoulli', 'logistic', 'convolutional', 'pooling']
        self.valid_layer_types['intermediate'] = ['rbm_gaussian_bernoulli', 'rbm_bernoulli_bernoulli', 'convolutional', 'pooling']
        self.valid_layer_types['last'] = ['rbm_gaussian_bernoulli', 'rbm_bernoulli_bernoulli', 'logistic']
        self.num_layers = num_layers
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
        if type(nn_weight2) is not Neural_Network_Weight:
            print "argument must be of type Neural_Network_Weight... instead of type", type(nn_weight2), "Exiting now..."
            sys.exit()
        return_val = 0
        for key in self.bias.keys():
            if key in excluded_keys['bias']:
                continue
            return_val += np.sum(self.bias[key] * nn_weight2.bias[key])
        for key in self.weights.keys():
            if key in excluded_keys['weights']:
                continue
            return_val += np.sum(self.weights[key] * nn_weight2.weights[key])
        return return_val
    def __str__(self):
        string = ""
        for key in self.bias.keys():
            string = string + "bias key " + key + "\n"
            string = string + str(self.bias[key]) + "\n"
        for key in self.weights.keys():
            string = string + "weight key " + key + "\n"
            string = string + str(self.weights[key]) + "\n"
        return string
    def print_statistics(self):
        for key in self.bias.keys():
            print "min of bias[" + key + "] is", np.min(self.bias[key]) 
            print "max of bias[" + key + "] is", np.max(self.bias[key])
            print "mean of bias[" + key + "] is", np.mean(self.bias[key])
            print "var of bias[" + key + "] is", np.var(self.bias[key]), "\n"
        for key in self.weights.keys():
            print "min of weights[" + key + "] is", np.min(self.weights[key]) 
            print "max of weights[" + key + "] is", np.max(self.weights[key])
            print "mean of weights[" + key + "] is", np.mean(self.weights[key])
            print "var of weights[" + key + "] is", np.var(self.weights[key]), "\n"
    def norm(self, excluded_keys = {'bias': [], 'weights': []}):
        squared_sum = 0
        for key in self.bias.keys():
            if key in excluded_keys['bias']:
                continue
            squared_sum += np.sum(self.bias[key] ** 2)
        for key in self.weights.keys():
            if key in excluded_keys['weights']:
                continue  
            squared_sum += np.sum(self.weights[key] ** 2)
        return np.sqrt(squared_sum)
    def max(self, excluded_keys = {'bias': [], 'weights': []}):
        max_val = -float('Inf')
        for key in self.bias.keys():
            if key in excluded_keys['bias']:
                continue
            max_val = max(np.max(self.bias[key]), max_val)
        for key in self.weights.keys():
            if key in excluded_keys['weights']:
                continue  
            max_val = max(np.max(self.weights[key]), max_val)
        return max_val
    def min(self, excluded_keys = {'bias': [], 'weights': []}):
        min_val = float('Inf')
        for key in self.bias.keys():
            if key in excluded_keys['bias']:
                continue
            min_val = min(np.min(self.bias[key]), min_val)
        for key in self.weights.keys():
            if key in excluded_keys['weights']:
                continue  
            min_val = min(np.min(self.weights[key]), min_val)
        return min_val
    def clip(self, clip_min, clip_max, excluded_keys = {'bias': [], 'weights': []}):
        nn_output = copy.deepcopy(self)
        for key in self.bias.keys():
            if key in excluded_keys['bias']:
                continue
            np.clip(self.bias[key], clip_min, clip_max, out=nn_output.bias[key])
        for key in self.weights.keys():
            if key in excluded_keys['weights']:
                continue  
            np.clip(self.weights[key], clip_min, clip_max, out=nn_output.weights[key])
        return nn_output
    def get_architecture(self):
        return [self.bias[str(layer_num)].size for layer_num in range(self.num_layers+1) ]
    def size(self, excluded_keys = {'bias': [], 'weights': []}):
        numel = 0
        for key in self.bias.keys():
            if key in excluded_keys['bias']:
                continue
            numel += self.bias[key].size
        for key in self.weights.keys():
            if key in excluded_keys['weights']:
                continue  
            numel += self.weights[key].size
        return numel
    def open_weights(self, weight_matrix_name): #completed
        #the weight file format is very specific, it contains the following variables:
        #weights01, weights12, weights23, ...
        #bias0, bias1, bias2, bias3, ....
        #weights01_type, weights12_type, weights23_type, etc...
        #optional variables:
        #num_layers
        #everything else will be ignored
        try:
            weight_dict = sp.loadmat(weight_matrix_name)
        except IOError:
            print "Unable to open", weight_matrix_name, "exiting now"
            sys.exit()
        if 'num_layers' in weight_dict:
            self.num_layers = weight_dict['num_layers'][0]
            if type(self.num_layers) is not int: #hack because write_weights() stores num_layers as [[num_layers]] 
                self.num_layers = self.num_layers[0]
        else: #count number of biases for num_layers
            self.num_layers = 0
            for layer_num in range(1,101): #maximum number of layers currently is set to 100
                if ''.join(['bias', str(layer_num)]) in weight_dict:
                    self.num_layers += 1
                else:
                    break
            if self.num_layers == 0:
                print "no layers found. Need at least one layer... Exiting now"
                sys.exit()
        try:
            self.bias['0'] = weight_dict['bias0']
        except KeyError:
            print "bias0 not found. bias0 must exist for", weight_matrix_name, "to be a valid weight file... Exiting now"
            sys.exit()
        for layer_num in range(1,self.num_layers+1): #changes weight layer type to ascii string, which is what we'll need for later functions
            weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
            bias_cur_layer = str(layer_num)
            self.weights[weight_cur_layer] = weight_dict[''.join(['weights', weight_cur_layer])]
            self.bias[bias_cur_layer] = weight_dict[''.join(['bias', bias_cur_layer])]  
            self.weight_type[weight_cur_layer] = weight_dict[''.join(['weights',weight_cur_layer,'_type'])][0].encode('ascii', 'ignore')
        del weight_dict
        self.check_weights()
    def init_random_weights(self, architecture, initial_bias_max, initial_bias_min, initial_weight_min, 
                           initial_weight_max, last_layer_logistic=True, seed=0): #completed, expensive, should be compiled
        np.random.seed(seed)
        self.num_layers = len(architecture) - 1
        initial_bias_range = initial_bias_max - initial_bias_min
        initial_weight_range = initial_weight_max - initial_weight_min
        self.bias['0'] = initial_bias_min + initial_bias_range * np.random.random_sample((1,architecture[0]))
        
        for layer_num in range(1,self.num_layers+1):
            weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
            bias_cur_layer = str(layer_num)
            #print "initializing weight layer", weight_cur_layer, "and bias layer", bias_cur_layer
            self.bias[bias_cur_layer] = initial_bias_min + initial_bias_range * np.random.random_sample((1,architecture[layer_num]))
            self.weights[weight_cur_layer]=(initial_weight_min + initial_weight_range * 
                                            np.random.random_sample( (architecture[layer_num-1],architecture[layer_num]) ))
            if layer_num == 0:
                self.weight_type[weight_cur_layer] = 'rbm_gaussian_bernoulli'
            elif layer_num == self.num_layers and last_layer_logistic == True:
                self.weight_type[weight_cur_layer] = 'logistic'
            else:
                self.weight_type[weight_cur_layer] = 'rbm_bernoulli_bernoulli'
        
        print "Finished Initializing Weights"
        self.check_weights()
    def init_zero_weights(self, architecture, last_layer_logistic=True, verbose=False):
        self.num_layers = len(architecture) - 1
        self.bias['0'] = np.zeros((1,architecture[0]))
        
        for layer_num in range(1,self.num_layers+1):
            weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
            bias_cur_layer = str(layer_num)
            #print "initializing weight layer", weight_cur_layer, "and bias layer", bias_cur_layer
            self.bias[bias_cur_layer] = np.zeros((1,architecture[layer_num]))
            self.weights[weight_cur_layer] = np.zeros( (architecture[layer_num-1],architecture[layer_num]) )
            if layer_num == 0:
                self.weight_type[weight_cur_layer] = 'rbm_gaussian_bernoulli'
            elif layer_num == self.num_layers and last_layer_logistic == True:
                self.weight_type[weight_cur_layer] = 'logistic'
            else:
                self.weight_type[weight_cur_layer] = 'rbm_bernoulli_bernoulli'
        if verbose:
            print "Finished Initializing Weights"
        self.check_weights(False)
    def check_weights(self, verbose=True): #need to check consistency of features with weights
        #checks weights to see if following conditions are true
        # *feature dimension equal to number of rows of first layer (if weights are stored in n_rows x n_cols)
        # *n_cols of (n-1)th layer == n_rows of nth layer
        # if only one layer, that weight layer type is logistic, gaussian_bernoulli or bernoulli_bernoulli
        # check is biases match weight values
        # if multiple layers, 0 to (n-1)th layer is gaussian bernoulli RBM or bernoulli bernoulli RBM and last layer is logistic regression
        
        #if below is true, not running in logistic regression mode, so first layer must be an RBM
        if verbose:
            print "Checking weights...",
        if self.num_layers > 1: 
            if self.weight_type['01'] not in self.valid_layer_types['intermediate']:
                print self.weight_type['01'], "is not valid layer type. Must be one of the following:", self.valid_layer_types['intermediate'], "...Exiting now"
                sys.exit()
        
        #check biases
        if self.bias['0'].shape[1] != self.weights['01'].shape[0]:
            print "Number of visible bias dimensions: ", self.bias['0'].shape[1],
            print " of layer 0 does not equal visible weight dimensions ", self.weights['01'].shape[0], "... Exiting now"
            sys.exit()
        if self.bias['1'].shape[1] != self.weights['01'].shape[1]:
            print "Number of hidden bias dimensions: ", self.weights['bias1'].shape[1],
            print " of layer 0 does not equal hidden weight dimensions ", self.weights['weights01'].shape[1], "... Exiting now"
            sys.exit()
        
        #intermediate layers need to have correct shape and RBM type
        for layer_num in range(1,self.num_layers-1): 
            weight_prev_layer = ''.join([str(layer_num-1),str(layer_num)])
            weight_cur_layer = ''.join([str(layer_num),str(layer_num+1)])
            bias_prev_layer = str(layer_num)
            bias_cur_layer = str(layer_num+1)
            #check shape
            if self.weights[weight_prev_layer].shape[1] != self.weights[weight_cur_layer].shape[0]:
                print "Dimensionality of", weight_prev_layer, "\b:", self.weights[weight_prev_layer].shape, "does not match dimensionality of", weight_cur_layer, "\b:",self.weights[weight_cur_layer].shape
                print "The second dimension of", weight_prev_layer, "must equal the first dimension of", weight_cur_layer
                sys.exit()
            #check RBM type
            if self.weight_type[weight_cur_layer] not in self.valid_layer_types['intermediate']:
                print self.weight_type[weight_cur_layer], "is not valid layer type. Must be one of the following:", self.valid_layer_types['intermediate'], "...Exiting now"
                sys.exit()
            #check biases
            if self.bias[bias_prev_layer].shape[1] != self.weights[weight_cur_layer].shape[0]:
                print "Number of visible bias dimensions:", self.bias[bias_prev_layer].shape[1], "of layer", weight_cur_layer, "does not equal visible weight dimensions:", self.weights[weight_cur_layer].shape[0]
                sys.exit()
            if self.bias[bias_cur_layer].shape[1] != self.weights[weight_cur_layer].shape[1]:
                print "Number of hidden bias dimensions:", self.bias[bias_cur_layer].shape[1],"of layer", weight_cur_layer, "does not equal hidden weight dimensions", self.weights[weight_cur_layer].shape[1]
                sys.exit()
        
        #check last layer
        layer_num = self.num_layers-1
        weight_prev_layer = ''.join([str(layer_num-1),str(layer_num)])
        weight_cur_layer = ''.join([str(layer_num),str(layer_num+1)])
        bias_prev_layer = str(layer_num)
        bias_cur_layer = str(layer_num+1)
        #check if last layer is of type logistic
        if self.weight_type[weight_cur_layer] not in self.valid_layer_types['last']:
            print self.weight_type[weight_cur_layer], " is not valid type for last layer.", 
            print "Must be one of the following:", self.valid_layer_types['last'], "...Exiting now"
            sys.exit()
        #check shape if hidden layer is used
        if self.num_layers > 1:
            if self.weights[weight_prev_layer].shape[1] != self.weights[weight_cur_layer].shape[0]:
                print "Dimensionality of", weight_prev_layer, "\b:", self.weights[weight_prev_layer].shape, "does not match dimensionality of", weight_cur_layer, "\b:",self.weights[weight_cur_layer].shape
                print "The second dimension of", weight_prev_layer, "must equal the first dimension of", weight_cur_layer
                sys.exit()
            #check biases
            if self.bias[bias_prev_layer].shape[1] != self.weights[weight_cur_layer].shape[0]:
                print "Number of visible bias dimensions:", self.weights[bias_prev_layer].shape[1], "of layer", weight_cur_layer, "does not equal visible weight dimensions:", self.weights[weight_cur_layer].shape[0]
                sys.exit()
            if self.bias[bias_cur_layer].shape[1] != self.weights[weight_cur_layer].shape[1]:
                print "Number of hidden bias dimensions:", self.weights[bias_cur_layer].shape[1],"of layer", weight_cur_layer, "does not equal hidden weight dimensions", self.weights[weight_cur_layer].shape[1]
                sys.exit()
        if verbose:
            print "seems copacetic"
    def write_weights(self, output_name): #completed
        weight_dict = dict()
        weight_dict['num_layers'] = self.num_layers
        weight_dict['bias0'] = self.bias['0']
        for layer_num in range(1, self.num_layers+1):
            bias_cur_layer = str(layer_num)
            weight_cur_layer = ''.join([str(layer_num-1), str(layer_num)])
            weight_dict[''.join(['bias', bias_cur_layer])] = self.bias[bias_cur_layer]
            weight_dict[''.join(['weights', weight_cur_layer])] = self.weights[weight_cur_layer]
            weight_dict[''.join(['weights', weight_cur_layer, '_type'])] = self.weight_type[weight_cur_layer]
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
        for key in self.bias.keys():
            nn_output.bias[key] = -self.bias[key]
        for key in self.weights.keys():
            nn_output.weights[key] = -self.weights[key]
        return nn_output
    def __add__(self,addend):
        nn_output = copy.deepcopy(self)
        if type(addend) is Neural_Network_Weight:
            if self.get_architecture() != addend.get_architecture():
                print "Neural net models do not match... Exiting now"
                sys.exit()
            
            for key in self.bias.keys():
                nn_output.bias[key] = self.bias[key] + addend.bias[key]
            for key in self.weights.keys():
                nn_output.weights[key] = self.weights[key] + addend.weights[key]
            return nn_output
        #otherwise type is scalar
        addend = float(addend)
        for key in self.bias.keys():
            nn_output.bias[key] = self.bias[key] + addend
        for key in self.weights.keys():
            nn_output.weights[key] = self.weights[key] + addend
        return nn_output
        
    def __sub__(self,subtrahend):
        nn_output = copy.deepcopy(self)
        if type(subtrahend) is Neural_Network_Weight:
            if self.get_architecture() != subtrahend.get_architecture():
                print "Neural net models do not match... Exiting now"
                sys.exit()
            
            for key in self.bias.keys():
                nn_output.bias[key] = self.bias[key] - subtrahend.bias[key]
            for key in self.weights.keys():
                nn_output.weights[key] = self.weights[key] - subtrahend.weights[key]
            return nn_output
        #otherwise type is scalar
        subtrahend = float(subtrahend)
        for key in self.bias.keys():
            nn_output.bias[key] = self.bias[key] - subtrahend
        for key in self.weights.keys():
            nn_output.weights[key] = self.weights[key] - subtrahend
        return nn_output
    def __mul__(self, multiplier):
        #if type(scalar) is not float and type(scalar) is not int:
        #    print "__mul__ must be by a float or int. Instead it is type", type(scalar), "Exiting now"
        #    sys.exit()
        nn_output = copy.deepcopy(self)
        if type(multiplier) is Neural_Network_Weight:
            for key in self.bias.keys():
                nn_output.bias[key] = self.bias[key] * multiplier.bias[key]
            for key in self.weights.keys():
                nn_output.weights[key] = self.weights[key] * multiplier.weights[key]
            return nn_output
        #otherwise scalar type
        multiplier = float(multiplier)
        
        for key in self.bias.keys():
            nn_output.bias[key] = self.bias[key] * multiplier
        for key in self.weights.keys():
            nn_output.weights[key] = self.weights[key] * multiplier
        return nn_output
    def __div__(self, divisor):
        #if type(scalar) is not float and type(scalar) is not int:
        #    print "Divide must be by a float or int. Instead it is type", type(scalar), "Exiting now"
        #    sys.exit()
        nn_output = copy.deepcopy(self)
        if type(divisor) is Neural_Network_Weight:
            for key in self.bias.keys():
                nn_output.bias[key] = self.bias[key] / divisor.bias[key]
            for key in self.weights.keys():
                nn_output.weights[key] = self.weights[key] / divisor.weights[key]
            return nn_output
        #otherwise scalar type
        divisor = float(divisor)
        
        for key in self.bias.keys():
            nn_output.bias[key] = self.bias[key] / divisor
        for key in self.weights.keys():
            nn_output.weights[key] = self.weights[key] / divisor
        return nn_output
    def __iadd__(self, nn_weight2):
        if type(nn_weight2) is not Neural_Network_Weight:
            print "argument must be of type Neural_Network_Weight... instead of type", type(nn_weight2), "Exiting now..."
            sys.exit()
        if self.get_architecture() != nn_weight2.get_architecture():
            print "Neural net models do not match... Exiting now"
            sys.exit()

        for key in self.bias.keys():
            self.bias[key] += nn_weight2.bias[key]
        for key in self.weights.keys():
            self.weights[key] += nn_weight2.weights[key]
        return self
    def __isub__(self, nn_weight2):
        if type(nn_weight2) is not Neural_Network_Weight:
            print "argument must be of type Neural_Network_Weight... instead of type", type(nn_weight2), "Exiting now..."
            sys.exit()
        if self.get_architecture() != nn_weight2.get_architecture():
            print "Neural net models do not match... Exiting now"
            sys.exit()

        for key in self.bias.keys():
            self.bias[key] -= nn_weight2.bias[key]
        for key in self.weights.keys():
            self.weights[key] -= nn_weight2.weights[key]
        return self
    def __imul__(self, scalar):
        #if type(scalar) is not float and type(scalar) is not int:
        #    print "__imul__ must be by a float or int. Instead it is type", type(scalar), "Exiting now"
        #    sys.exit()
        scalar = float(scalar)
        for key in self.bias.keys():
            self.bias[key] *= scalar
        for key in self.weights.keys():
            self.weights[key] *= scalar
        return self
    def __idiv__(self, scalar):
        scalar = float(scalar)
        for key in self.bias.keys():
            self.bias[key] /= scalar
        for key in self.weights.keys():
            self.weights[key] /= scalar
        return self
    def __pow__(self, scalar):
        if scalar == 2:
            return self * self
        scalar = float(scalar)
        nn_output = copy.deepcopy(self)
        for key in self.bias.keys():
            nn_output.bias[key] = self.bias[key] ** scalar
        for key in self.weights.keys():
            nn_output.weights[key] = self.weights[key] ** scalar
        return nn_output
    def __copy__(self):
        return Neural_Network_Weight(self.num_layers, self.weights, self.bias, self.weight_type)
    def __deepcopy__(self, memo):
        return Neural_Network_Weight(copy.deepcopy(self.num_layers, memo), copy.deepcopy(self.weights,memo), 
                                     copy.deepcopy(self.bias,memo), copy.deepcopy(self.weight_type,memo))

class Neural_Network(object, Vector_Math):
    #features are stored in format ndata x nvis
    #weights are stored as nvis x nhid at feature level
    #biases are stored as 1 x nhid
    #rbm_type is either gaussian_bernoulli, bernoulli_bernoulli, notrbm_logistic
    def __init__(self, config_dictionary): #completed
        #variables for Neural Network: feature_file_name(read from)
        #required_variables - required variables for running system
        #all_variables - all valid variables for each type
        self.feature_file_name = self.default_variable_define(config_dictionary, 'feature_file_name', arg_type='string')
        self.features = self.read_feature_file()
        self.model = Neural_Network_Weight()
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
            return sp.loadmat(self.feature_file_name)['features'] #in MATLAB format
        except IOError:
            print "Unable to open ", self.feature_file_name, "... Exiting now"
            sys.exit()
    def read_label_file(self): #completed
        try:
            return sp.loadmat(self.label_file_name)['labels'] #in MATLAB format
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
        print "Checking labels..."
        if len(self.labels.shape) != 1 and ((len(self.labels.shape) == 2 and self.labels.shape[1] != 1) or len(self.labels.shape) > 2):
            print "labels need to be in (n_samples) or (n_samples,1) format and the shape of labels is ", self.labels.shape, "... Exiting now"
            sys.exit()
        if self.labels.size != self.features.shape[0]:
            print "Number of examples in feature file: ", self.features.shape[0], " does not equal size of label file, ", self.labels.size, "... Exiting now"
            sys.exit()
        if  [i for i in np.unique(self.labels)] != range(np.max(self.labels)+1):
            print "Labels need to be in the form 0,1,2,....,n,... Exiting now"
            sys.exit()
        label_counts = np.bincount(np.ravel(self.labels)) #[self.labels.count(x) for x in range(np.max(self.labels)+1)]
        print "distribution of labels is:"
        for x in range(len(label_counts)):
            print "#", x, "\b's:", label_counts[x]            
        print "labels seem copacetic"
    def forward_layer(self, inputs, weights, biases, weight_type): #completed
        if weight_type == 'logistic':
            return self.softmax(self.weight_matrix_multiply(inputs, weights, biases))
        elif weight_type == 'rbm_gaussian_bernoulli' or weight_type == 'rbm_bernoulli_bernoulli':
            return self.sigmoid(self.weight_matrix_multiply(inputs, weights, biases))
        #added to test finite differences calculation for pearlmutter forward pass
        elif weight_type == 'linear':
            return self.weight_matrix_multiply(inputs, weights, biases)
        else:
            print "weight_type", weight_type, "is not a valid layer type.",
            print "Valid layer types are", self.model.valid_layer_types,"Exiting now..."
            sys.exit()
    def forward_pass_linear(self, inputs, verbose=True, model=None):
        #to test finite differences calculation for pearlmutter forward pass, just like forward pass, except it spits linear outputs
        if model == None:
            model = self.model 
        cur_layer = inputs
        for layer_num in range(1,model.num_layers):
            if verbose:
                print "At layer", layer_num, "of", model.num_layers
            weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
            bias_cur_layer = str(layer_num)
            cur_layer = self.forward_layer(cur_layer, model.weights[weight_cur_layer], 
                                           model.bias[bias_cur_layer], model.weight_type[weight_cur_layer])
        
        weight_cur_layer = ''.join([str(model.num_layers-1),str(model.num_layers)])
        bias_cur_layer = str(model.num_layers)
        cur_layer = self.forward_layer(cur_layer, model.weights[weight_cur_layer], 
                                           model.bias[bias_cur_layer], 'linear')
        return cur_layer
    def forward_pass(self, inputs, verbose=True, model=None): #completed
        """forward pass each layer starting with feature level"""
        if model == None:
            model = self.model 
        cur_layer = inputs
        for layer_num in range(1,model.num_layers+1):
            if verbose:
                print "At layer", layer_num, "of", model.num_layers
            weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
            bias_cur_layer = str(layer_num)
            cur_layer = self.forward_layer(cur_layer, model.weights[weight_cur_layer], 
                                           model.bias[bias_cur_layer], model.weight_type[weight_cur_layer])
        return cur_layer
    def calculate_cross_entropy(self, output, labels): #completed, expensive, should be compiled
        return -np.sum(np.log([max(output.item((x,labels[x])),1E-12) for x in range(labels.size)]))
    def calculate_classification_accuracy(self, output, labels): #completed, possibly expensive
        prediction = output.argmax(axis=1).reshape(labels.shape)
        classification_accuracy = sum(prediction == labels) / float(labels.size)
        return classification_accuracy[0]
                
class NN_Tester(Neural_Network): #completed
    def __init__(self, config_dictionary): #completed
        """runs DNN tester soup to nuts.
        variables are
        feature_file_name - name of feature file to load from
        weight_matrix_name - initial weight matrix to load
        output_name - output predictions
        label_file_name - label file to check accuracy
        required are feature_file_name, weight_matrix_name, and output_name"""
        self.mode = 'test'
        super(NN_Tester,self).__init__(config_dictionary)
        self.check_keys(config_dictionary)
        
        self.weight_matrix_name = self.default_variable_define(config_dictionary, 'weight_matrix_name', arg_type='string')
        self.model.open_weights(self.weight_matrix_name)
        self.label_file_name = self.default_variable_define(config_dictionary, 'label_file_name', arg_type='string',error_string="No label_file_name defined, just running forward pass",exit_if_no_default=False)
        if self.label_file_name != None:
            self.labels = self.read_label_file()
            self.check_labels()
        else:
            del self.label_file_name
        self.dump_config_vals()
        self.classify()
        self.write_posterior_prob_file()
    def classify(self): #completed
        self.posterior_probs = self.forward_pass(self.features)
        try:
            avg_cross_entropy = self.calculate_cross_entropy(self.posterior_probs, self.labels) / self.labels.size
            print "Average cross-entropy is", avg_cross_entropy
            print "Classification accuracy is", self.calculate_classification_accuracy(self.posterior_probs, self.labels) * 100, "\b%"
        except AttributeError:
            print "no labels given, so skipping classification statistics"    
    def write_posterior_prob_file(self): #completed
        try:
            print "Writing to", self.output_name
            sp.savemat(self.output_name,{'targets' : self.posterior_probs}, oned_as='column') #output name should have .mat extension
        except IOError:
            print "Unable to write to ", self.output_name, "... Exiting now"
            sys.exit()
        
class NN_Trainer(Neural_Network):
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
        super(NN_Trainer,self).__init__(config_dictionary)
        self.num_training_examples = self.features.shape[0]
        self.check_keys(config_dictionary)
        #read label file
        self.label_file_name = self.default_variable_define(config_dictionary, 'label_file_name', arg_type='string', error_string="No label_file_name defined, can only do pretraining",exit_if_no_default=False)
        if self.label_file_name != None:
            self.labels = self.read_label_file()
            self.check_labels()
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
    def train(self): #completed
        if self.do_pretrain:
            self.pretrain()
        if self.do_backprop:
            if self.backprop_method == 'steepest_descent':
                self.backprop_steepest_descent()
            elif self.backprop_method == 'conjugate_gradient':
                self.backprop_conjugate_gradient()
            elif self.backprop_method == 'krylov_subspace':
                self.backprop_krylov_subspace()
            elif self.backprop_method == 'truncated_newton':
                self.backprop_truncated_newton()
            else:
                print self.backprop_method, "is not a valid type of backprop (supported ones are steepest_descent, conjugate_gradient, krylov_subspace, and truncated_newton... Exiting now"
                sys.exit()
        self.model.write_weights(self.output_name)
    def calculate_loss(self, inputs, labels, model = None, l2_regularization_const = None):
        #differs from calculate_cross_entropy in that it also allows for regularization term
        batch_size = inputs.shape[0]
        if batch_size == inputs.size:
            batch_size = 1
        if model == None:
            model = self.model
        if l2_regularization_const == None:
            l2_regularization_const = self.l2_regularization_const
        excluded_keys = {'bias':['0'], 'weights':[]}
        outputs = self.forward_pass(inputs, verbose=False, model = model)
        if self.l2_regularization_const == 0.0:
            return self.calculate_cross_entropy(outputs, labels)
        else:
            return self.calculate_cross_entropy(outputs, labels) + (model.norm(excluded_keys) ** 2) * l2_regularization_const / 2. * batch_size#model.dot(model, excluded_keys) * self.l2_regularization_const
    #pretraining functions
    def backward_layer(self, hiddens, weights, biases, weight_type): #completed, transpose expensive, should be compiled
        if weight_type == 'rbm_gaussian_bernoulli':
            return self.weight_matrix_multiply(hiddens, np.transpose(weights), biases)
        else: #rbm_type is bernoulli
            return self.sigmoid(self.weight_matrix_multiply(hiddens, np.transpose(weights), biases))
    def pretrain(self): #completed, weight updates expensive, should be compiled
        print "starting pretraining"
        learning_rate_index = 0;
        for layer_num in range(len(self.pretrain_iterations)):
            print "pretraining rbm", layer_num+1, "of", len(self.pretrain_iterations)
            for iteration in range(self.pretrain_iterations[layer_num]):
                print "at iteration", iteration+1, "of", self.pretrain_iterations[layer_num]
                batch_index = 0
                end_index = 0
                reconstruction_error = 0
                while end_index < self.num_training_examples: #run through batches
                    per_done = float(batch_index)/self.num_training_examples*100
                    sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
                    end_index = min(batch_index+self.pretrain_batch_size,self.num_training_examples)
                    batch_size = end_index - batch_index
                    inputs = self.features[batch_index:end_index]
                    for idx in range(layer_num): #propagate to current pre-training layer
                        bias_cur_layer = str(idx+1)
                        weight_cur_layer = ''.join([str(idx), str(idx+1)])
                        inputs = self.forward_layer(inputs, self.model.weights[weight_cur_layer], 
                                                    self.model.bias[bias_cur_layer], self.model.weight_type[weight_cur_layer])
                
                    bias_cur_layer  = str(layer_num+1)
                    bias_prev_layer = str(layer_num)
                    weight_cur_layer = ''.join([str(layer_num), str(layer_num+1)])
                                        
                    hiddens = self.forward_layer(inputs, self.model.weights[weight_cur_layer], self.model.bias[bias_cur_layer], self.model.weight_type[weight_cur_layer]) #ndata x nhid
                    reconstruction = self.backward_layer(hiddens, self.model.weights[weight_cur_layer], self.model.bias[bias_prev_layer], self.model.weight_type[weight_cur_layer]) #ndata x nvis
                    reconstruction_hiddens = self.forward_layer(reconstruction, self.model.weights[weight_cur_layer], self.model.bias[bias_cur_layer], self.model.weight_type[weight_cur_layer]) #ndata x nhid
                
                    #update weights
                    weight_update =  np.dot(np.transpose(reconstruction),reconstruction_hiddens) - np.dot(np.transpose(inputs),hiddens) # inputs: [batch_size * n_dim], hiddens: [batch_size * n_hids]
                    vis_bias_update =  np.sum(reconstruction, axis=0) - np.sum(inputs, axis=0)
                    hid_bias_update =  np.sum(reconstruction_hiddens, axis=0) - np.sum(hiddens, axis=0)
                    self.model.weights[weight_cur_layer] -= self.pretrain_learning_rate[learning_rate_index] / batch_size * weight_update
                    self.model.bias[bias_prev_layer] -= self.pretrain_learning_rate[learning_rate_index] / batch_size * vis_bias_update
                    self.model.bias[bias_cur_layer] -= self.pretrain_learning_rate[learning_rate_index] / batch_size * hid_bias_update
                    
                    reconstruction_error += np.sum((inputs - reconstruction) * (inputs - reconstruction))
                    batch_index += self.pretrain_batch_size
                sys.stdout.write("\r100.0% done \r")
                print "squared reconstuction error is", reconstruction_error
                if self.save_each_epoch:
                    self.model.write_weights(''.join([self.output_name, '_pretrain_rbm_', str(layer_num+1), '_iter_', str(iteration+1)]))
                learning_rate_index += 1
    #fine-tuning/backprop functions
    #currently implemented are stochastic/steepest descent
    #and conjugate gradient methods
    def forward_first_order_methods(self, inputs, model = None): #completed
        #returns hidden values for each layer, needed for steepest descent and conjugate gradient methods
        if model == None:
            model = self.model
        hiddens = dict()
        hiddens[0] = inputs
        for layer_num in range(1,model.num_layers+1): #will need for steepest descent for first direction
            weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
            bias_cur_layer = str(layer_num)
            hiddens[layer_num] = self.forward_layer(hiddens[layer_num-1], model.weights[weight_cur_layer], 
                                                    model.bias[bias_cur_layer], model.weight_type[weight_cur_layer] )
        return hiddens    
    def backprop_steepest_descent(self): #need to test regularization
        print "starting backprop using steepest descent"
        print "Number of layers is", self.model.num_layers
        
        cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.model)
        print "cross-entropy before steepest descent is", cross_entropy
        if self.l2_regularization_const > 0.0:
            print "regularized loss is", loss
        print "number correctly classified is", num_correct, "of", num_examples

        for epoch_num in range(len(self.steepest_learning_rate)):
            print "At epoch", epoch_num+1, "of", len(self.steepest_learning_rate), "with learning rate", self.steepest_learning_rate[epoch_num]
            batch_index = 0
            end_index = 0
            while end_index < self.num_training_examples: #run through the batches
                per_done = float(batch_index)/self.num_training_examples*100
                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
                end_index = min(batch_index+self.backprop_batch_size,self.num_training_examples)
                batch_size = end_index - batch_index
                hiddens = self.forward_first_order_methods(self.features[batch_index:end_index])
                #calculating negative gradient of log softmax
                weight_vec = -hiddens[self.model.num_layers] #batchsize x n_outputs
                for label_index in range(batch_index,end_index):
                    data_index = label_index - batch_index
                    weight_vec[data_index, self.labels[label_index]] += 1
                #averaging batches
                bias_update = sum(weight_vec)
                weight_update = np.dot(np.transpose(hiddens[self.model.num_layers-1]), weight_vec)
                #I don't use calculate_gradient because structure allows me to store only one layer of weights
                
                for layer_num in range(self.model.num_layers-1,0,-1):
                    weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
                    weight_next_layer = ''.join([str(layer_num),str(layer_num+1)])
                    bias_cur_layer = str(layer_num)
                    bias_next_layer = str(layer_num+1)
                    weight_vec = np.dot(weight_vec, np.transpose(self.model.weights[weight_next_layer])) * hiddens[layer_num] * (1-hiddens[layer_num]) #n_hid x n_out * (batchsize x n_out), do the biases get involved in this calculation???
                    
                    self.model.weights[weight_next_layer] += self.steepest_learning_rate[epoch_num] / batch_size * (weight_update - self.l2_regularization_const * self.model.weights[weight_next_layer])
                    self.model.bias[bias_next_layer][0] += self.steepest_learning_rate[epoch_num] / batch_size * (bias_update - self.l2_regularization_const * self.model.bias[bias_next_layer][0])
                    bias_update = sum(weight_vec)
                    weight_update = np.dot(np.transpose(hiddens[layer_num-1]), weight_vec)
                    
                #do final weight_update
                self.model.weights[weight_cur_layer] += self.steepest_learning_rate[epoch_num] / batch_size * (weight_update - self.l2_regularization_const * self.model.weights[weight_cur_layer])
                self.model.bias[bias_cur_layer][0] += self.steepest_learning_rate[epoch_num] / batch_size * (bias_update - self.l2_regularization_const * self.model.bias[bias_cur_layer][0])

                batch_index += self.backprop_batch_size
            sys.stdout.write("\r100.0% done \r"), sys.stdout.flush()
            
            cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.model)
            print "cross-entropy at the end of the epoch is", cross_entropy
            if self.l2_regularization_const > 0.0:
                print "regularized loss is", loss
            print "number correctly classified is", num_correct, "of", num_examples
            
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))
                
    def backprop_conjugate_gradient(self): #Running... need preconditioners
        #in this framework "points" are self.weights
        #will also need to store CG-direction, which will be in dictionary conj_grad_dir
        print "Starting backprop using conjugate gradient"
        print "Number of layers is", self.model.num_layers
        
        cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.model)
        print "cross-entropy before conjugate gradient is", cross_entropy
        if self.l2_regularization_const > 0.0:
            print "regularized loss is", loss
        print "number correctly classified is", num_correct, "of", num_examples
        #we have three new gradients now: conjugate_gradient_dir, old_gradient, and new_gradient
        excluded_keys = {'bias':['0'], 'weights':[]} #will have to change this later
        init_step_num = 0
        step_size = 0
        for epoch_num in range(self.num_epochs):
            print "Epoch", epoch_num+1, "of", self.num_epochs
            num_conj_dir_switches = 0
            batch_index = 0
            end_index = 0
            line_search_failures = 0
            while end_index < self.num_training_examples: #run through the batches
                per_done = float(batch_index)/self.num_training_examples*100
                print "\r                                                                                                         \r", #clear line
                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
                end_index = min(batch_index+self.backprop_batch_size,self.num_training_examples)
                batch_inputs = self.features[batch_index:end_index]
                batch_labels = self.labels[batch_index:end_index]
                
                ########## perform conjugate gradient on the batch ########################
                failed_line_search = False
                conj_grad_dir = -self.calculate_gradient(batch_inputs, batch_labels) #steepest descent for first direction
                old_gradient = copy.deepcopy(conj_grad_dir)
                new_gradient = copy.deepcopy(conj_grad_dir)
                init_step_size = step_size * init_step_num / (conj_grad_dir.norm(excluded_keys) ** 2)
                
                for _ in range(self.conjugate_max_iterations):
                    step_size = self.line_search(batch_inputs, batch_labels, conj_grad_dir, max_line_searches=self.num_line_searches, 
                                                 init_step_size=init_step_size, max_step_size=3.0, verbose=True)
                    if step_size > 0: #line search did not fail
                        #update weights if successful
                        self.model += conj_grad_dir * step_size
                        failed_line_search = False
                        #update search direction
                        new_gradient = self.calculate_gradient(batch_inputs, batch_labels)
                        init_step_num = abs(old_gradient.dot(conj_grad_dir,excluded_keys)) #since we know conj_grad_dir is a descent dir
                        conj_grad_dir = self.calculate_conjugate_gradient_direction(batch_inputs, batch_labels, old_gradient, new_gradient, 
                                                                                    conj_grad_dir, const_type=self.conjugate_const_type)
                        init_step_size = step_size * init_step_num / abs(new_gradient.dot(conj_grad_dir, excluded_keys))
                        old_gradient.clear()
                        old_gradient = copy.deepcopy(new_gradient)
                        new_gradient.clear()
                        if old_gradient.dot(conj_grad_dir, excluded_keys) > 0: #conjugate gradient direction not a descent direction, switching to steepest descent
                            sys.stdout.write("\rCalculated conjugate direction not a descent direction, switching direction to negative gradient"), sys.stdout.flush()
                            num_conj_dir_switches += 1
                            conj_grad_dir = -self.calculate_gradient(batch_inputs, batch_labels)
                            init_step_size = step_size * init_step_num / (conj_grad_dir.norm(excluded_keys) ** 2)
                            old_gradient = copy.deepcopy(conj_grad_dir)
                    else: #line search failed
                        line_search_failures += 1
                        if failed_line_search: #failed line search twice in a row, so bail
                            sys.stdout.write("\rline search failed twice. Moving to next batch...\r"), sys.stdout.flush()
                            break
                        sys.stdout.write("\rline search failed...\r"), sys.stdout.flush()
                        failed_line_search = True
                        conj_grad_dir = -self.calculate_gradient(batch_inputs, batch_labels)
                        init_step_size = 0.0
                        old_gradient = conj_grad_dir
                ###########end conjugate gradient batch ####################################
                batch_index += self.backprop_batch_size
            sys.stdout.write("\r100.0% done \r")
            print "number of failed line search in this epoch is", line_search_failures
            print "number of times conjugate direction was not a descent direction is", num_conj_dir_switches
            
            cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.model)
            print "cross-entropy at the end of the epoch is", cross_entropy
            if self.l2_regularization_const > 0.0:
                print "regularized loss is", loss
            print "number correctly classified is", num_correct, "of", num_examples
            
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))
    def backprop_truncated_newton(self):
        print "Starting backprop using truncated newton"
        print "Number of layers is", self.model.num_layers
        
        cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.model)
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
            
            while end_index < self.num_training_examples: #run through the batches
                per_done = float(batch_index)/self.num_training_examples*100
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\r%.1f%% done " % per_done), sys.stdout.flush()
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\rdamping factor is %f\r" % damping_factor), sys.stdout.flush()
                end_index = min(batch_index+self.backprop_batch_size,self.num_training_examples)
                batch_inputs = self.features[batch_index:end_index]
                batch_labels = self.labels[batch_index:end_index]
                batch_size = batch_inputs.shape[0]
                
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("\rcalculating gradient\r"), sys.stdout.flush()
                gradient = self.calculate_gradient(batch_inputs, batch_labels, check_gradient=False, model=self.model)
                
                old_loss = self.calculate_loss(batch_inputs, batch_labels, model=self.model) #self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False, model=self.model), batch_labels)
                
                if self.use_fisher_preconditioner:
                    sys.stdout.write("\r                                                                \r")
                    sys.stdout.write("calculating diagonal Fisher matrix for preconditioner"), sys.stdout.flush()
                    
                    preconditioner = self.calculate_fisher_diag_matrix(batch_inputs, batch_labels, False, self.model, l2_regularization_const = 0.0)
                    # add regularization
                    #preconditioner = preconditioner + alpha / preconditioner.size(excluded_keys) * self.model.norm(excluded_keys) ** 2
                    preconditioner = (preconditioner + self.l2_regularization_const + damping_factor) ** (3./4.)
                    preconditioner = preconditioner.clip(preconditioner.max(excluded_keys) * self.fisher_preconditioner_floor_val, float("Inf"))
                model_update, model_vals = self.conjugate_gradient(batch_inputs, batch_labels, self.truncated_newton_num_cg_epochs, model=self.model, 
                                                                   damping_factor=damping_factor, preconditioner=preconditioner, gradient=gradient, 
                                                                   second_order_type=self.second_order_matrix, init_search_direction=model_update,
                                                                   verbose = False)
                model_den = model_vals[-1] #- model_vals[0]
                
                self.model += model_update
                new_loss = self.calculate_loss(batch_inputs, batch_labels, model=self.model)  #self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False, model=self.model), batch_labels)
                model_num = (new_loss - old_loss) / batch_size
                sys.stdout.write("\r                                                                      \r") #clear line
                print "model ratio is", model_num / model_den,
                if model_num / model_den < 0.25:
                    damping_factor *= 1.5
                elif model_num / model_den > 0.75:
                    damping_factor *= 2./3.
                batch_index += self.backprop_batch_size
                
                cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
                if self.l2_regularization_const > 0.0:
                    print "regularized loss is", loss
                print "number correctly classified is", num_correct, "of", num_examples
                
            sys.stdout.write("\r100.0% done \r")
            
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)]))
    def backprop_krylov_subspace(self):
        #does backprop using krylov subspace
        excluded_keys = {'bias':['0'], 'weights':[]} #will have to change this later
        print "Starting backprop using krylov subspace descent"
        print "Number of layers is", self.model.num_layers
        
        cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.model)
        print "cross-entropy before krylov subspace is", cross_entropy
        if self.l2_regularization_const > 0.0:
            print "regularized loss is", loss
        print "number correctly classified is", num_correct, "of", num_examples
        
        prev_direction = Neural_Network_Weight(self.model.num_layers) #copy.deepcopy(self.model) * 0
        #print self.model.get_architecture()
        prev_direction.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True)
        prev_direction.bias['0'][0][0] = 1
        sub_batch_start_perc = 0.0
        preconditioner = None
        for epoch_num in range(self.num_epochs):
            print "Epoch", epoch_num + 1, "of", self.num_epochs
            batch_index = 0
            end_index = 0
            while end_index < self.num_training_examples: #run through the batches
                #per_done = float(batch_index)/self.num_training_examples*100
                #sys.stdout.write("\r%.1f%% done" % per_done), sys.stdout.flush()
                end_index = min(batch_index+self.backprop_batch_size,self.num_training_examples)
                batch_size = end_index - batch_index 
                batch_inputs = self.features[batch_index:end_index] #:batch_index+1]
                batch_labels = self.labels[batch_index:end_index] #:batch_index+1]
                krylov_start_index = int(sub_batch_start_perc * batch_size)
                bfgs_start_index = int((sub_batch_start_perc + 1.0 / self.krylov_num_batch_splits) * batch_size)
                bfgs_end_index = int((sub_batch_start_perc + 2.0 / self.krylov_num_batch_splits) * batch_size)
                krylov_index = [batch_index + x % batch_size for x in range(krylov_start_index, bfgs_start_index)] #[batch_index]
                bfgs_index = [batch_index + x % batch_size for x in range(bfgs_start_index, bfgs_end_index)]
                #print "krylov_index:", krylov_index
                #print "bfgs_index:", bfgs_index
                #print "krylov_start_index:", krylov_start_index, "bfgs_start_index:", bfgs_start_index, "bfgs_end_index", bfgs_end_index
                sys.stdout.write("\r                                                                \r") #clear line
                sys.stdout.write("part 1/3: calculating gradient"), sys.stdout.flush()
                average_gradient = self.calculate_gradient(batch_inputs, batch_labels, False, self.model)
                
                #average_gradient.print_statistics()
                #need to fix the what indices the batches are taken from... will always be the same subset
                
                #average_gradient = self.calculate_gradient(krylov_batch_inputs, krylov_batch_labels, self.model) / (batch_size / self.krylov_num_batch_splits)
                if self.use_fisher_preconditioner:
                    sys.stdout.write("\r                                                                \r")
                    sys.stdout.write("part 1/3: calculating diagonal Fisher matrix for preconditioner"), sys.stdout.flush()
                    preconditioner = self.calculate_fisher_diag_matrix(batch_inputs, batch_labels, False, self.model)
                    #precon = (grad2 + mones(psize,1)*conv(lambda) + maskp*conv(weightcost)).^(3/4);
                    # add regularization
                    #preconditioner = preconditioner + alpha / preconditioner.size(excluded_keys) * self.model.norm(excluded_keys) ** 2
                    preconditioner = (preconditioner + self.l2_regularization_const) ** (3./4.)
                    preconditioner = preconditioner.clip(preconditioner.max(excluded_keys) * self.fisher_preconditioner_floor_val, float("Inf"))
                    #preconditioner.print_statistics()
                    #sys.exit()
                krylov_batch_inputs = self.features[[krylov_index]]
                krylov_batch_labels = self.labels[krylov_index]
                sys.stdout.write("\r                                                                \r")
                sys.stdout.write("part 2/3: calculating krylov basis"), sys.stdout.flush()
                krylov_basis = self.calculate_krylov_basis(krylov_batch_inputs, krylov_batch_labels, prev_direction, average_gradient, self.model, preconditioner) #, preconditioner = average_gradient ** 2)
                
                if self.krylov_use_hessian_preconditioner:
#                    U,singular_values,V = np.linalg.svd(krylov_basis['hessian'])
#                    np.clip(singular_values, np.max(singular_values) * self.krylov_eigenvalue_floor_const, float("Inf"), out=singular_values)
#                    projection_matrix = np.dot(U, np.diag(1. / np.sqrt(singular_values)))
#                    krylov_basis_copy = dict()
#                    for idx in range(self.krylov_num_directions+1):
#                        krylov_basis_copy[idx] = krylov_basis[0] * projection_matrix[0][idx]
#                        
#                    for krylov_idx in range(0,self.krylov_num_directions+1):
#                        for projection_idx in range(1,self.krylov_num_directions+1):
#                            krylov_basis_copy[krylov_idx] += krylov_basis[projection_idx] * projection_matrix[projection_idx][krylov_idx]
#                    del krylov_basis
#                    krylov_basis = krylov_basis_copy
                    eigenvalues, eigenvectors = np.linalg.eig(krylov_basis['hessian'])
                    eigenvalues = np.abs(eigenvalues)
                    np.clip(eigenvalues, np.max(eigenvalues) * self.krylov_eigenvalue_floor_const, float("Inf"), out=eigenvalues)
                    inv_hessian_cond = np.dot(np.dot(eigenvectors, np.diag(1./eigenvalues)),np.transpose(eigenvectors))
                    inv_chol_factor = sl.cholesky(inv_hessian_cond) #numpy version gives lower triangular, scipy gives ut
                    for basis_num in range(self.krylov_num_directions+1):
                        krylov_basis[basis_num] *= inv_chol_factor[basis_num][basis_num]
                        for basis_mix_idx in range(basis_num+1,self.krylov_num_directions+1):
                            krylov_basis[basis_num] += krylov_basis[basis_mix_idx] * inv_chol_factor[basis_num][basis_mix_idx]
                #some_grad = np.zeros(len(krylov_basis.keys())-1) #-1 for 'hessian' key
                #print some_grad
                #print some_grad.shape[0]
                #for dim in range(some_grad.shape[0]):
                #    some_grad[dim] = average_gradient.dot(krylov_basis[dim], excluded_keys) #check to see if GN matrix is PSD
                #print some_grad
                #sys.exit()
                bfgs_batch_inputs = self.features[[bfgs_index]]
                bfgs_batch_labels = self.labels[bfgs_index]
                sys.stdout.write("\r                                                                \r")
                sys.stdout.write("part 3/3: calculating mix of krylov basis using bfgs"), sys.stdout.flush()
                #step_size = sopt.fmin_bfgs(f=self.calculate_subspace_cross_entropy, x0=np.zeros(self.krylov_num_directions+1), 
                #                           fprime=self.calculate_subspace_gradient, args=(krylov_basis, bfgs_batch_inputs, bfgs_batch_labels, self.model), 
                #                           gtol=1E-5, norm=2, maxiter=self.krylov_num_bfgs_epochs)
                step_size = self.bfgs(bfgs_batch_inputs, bfgs_batch_labels, krylov_basis, self.krylov_num_bfgs_epochs)
                #print "returned step size is", step_size
                direction = krylov_basis[0] * step_size[0]
                for basis in range(1,len(step_size)):
                    direction += krylov_basis[basis] * step_size[basis]
                #print "printing direction statistics"
                #direction.print_statistics()
                #print "printing model statistics"
                #self.model.print_statistics()
                self.model += direction
                prev_direction = copy.deepcopy(direction)
                direction.clear()
                batch_index += self.backprop_batch_size
                cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.model)
                print "cross-entropy at the end of the epoch is", cross_entropy
                if self.l2_regularization_const > 0.0:
                    print "regularized loss is", loss
                print "number correctly classified is", num_correct, "of", num_examples
            #sub_batch_start_perc = (sub_batch_start_perc + 1.0 / self.krylov_num_batch_splits) % 1 #not sure if this is better, below line is what I used to get krylov results
#            sub_batch_start_perc = (sub_batch_start_perc + 2.0 / self.krylov_num_batch_splits) % 1
#            cross_entropy, num_correct, num_examples, loss = self.calculate_classification_statistics(self.features, self.labels, self.model)
#            
#            print "cross-entropy at the end of the epoch is", cross_entropy
#            if self.l2_regularization_const > 0.0:
#                print "regularized loss is", loss
#            print "number correctly classified is", num_correct, "of", num_examples
            if self.save_each_epoch:
                self.model.write_weights(''.join([self.output_name, '_epoch_', str(epoch_num+1)])) 
                
    def calculate_conjugate_gradient_direction(self, batch_inputs, batch_labels, old_gradient, new_gradient, #needs preconditioners, expensive, should be compiled
                                               current_conjugate_gradient_direction, const_type='polak-ribiere'):
        excluded_keys = {'bias':['0'], 'weights':[]} #will have to change this later
        if const_type == 'polak-ribiere' or const_type == 'polak-ribiere+':
            cg_const = new_gradient.dot(new_gradient - old_gradient, excluded_keys) / old_gradient.norm(excluded_keys)**2
            if const_type == 'polak-ribiere+':
                cg_const = max(cg_const, 0)
        elif const_type == 'fletcher-reeves':
            cg_const = new_gradient.norm(excluded_keys)**2 / old_gradient.norm(excluded_keys)**2
        elif const_type == 'hestenes-stiefel': #might run into numerical stability issues
            cg_const = new_gradient.dot(new_gradient - old_gradient, excluded_keys) / max(current_conjugate_gradient_direction.dot(new_gradient - old_gradient, excluded_keys), 1E-4)
        else:
            print const_type, "not recognized, the only valid methods of \'polak-ribiere\', \'fletcher-reeves\', \'polak-ribiere+\', \'hestenes-stiefel\'... Exiting now"
            sys.exit()
        
        return -new_gradient + current_conjugate_gradient_direction * cg_const
    def calculate_gradient(self, batch_inputs, batch_labels, check_gradient=False, model=None, l2_regularization_const = None, hiddens = None): #want to make this more general to handle arbitrary loss functions, structures, expensive, should be compiled
        #need to check regularization
        #calculate gradient with particular Neural Network model. If None is specified, will use current weights (i.e., self.model)
        excluded_keys = {'bias':['0'], 'weights':[]} #will have to change this later
        if model == None:
            model = self.model
        if l2_regularization_const == None:
            l2_regularization_const = self.l2_regularization_const
        #gradient = copy.deepcopy(model)
        batch_size = batch_inputs.shape[0]
        
        if hiddens == None:
            hiddens = self.forward_first_order_methods(batch_inputs, model)
        #derivative of log(cross-entropy softmax)
        weight_vec = hiddens[model.num_layers] #batchsize x n_outputs

        for index in range(batch_size):
            weight_vec[index, batch_labels[index]] -= 1
        gradient_weights = self.backward_pass(weight_vec, hiddens, model)
        
        if not check_gradient:
            if l2_regularization_const > 0.0:
                return gradient_weights / batch_size  + model * l2_regularization_const
            return gradient_weights / batch_size
        
        ### below block checks gradient... only to be used if you think the gradient is incorrectly calculated ##############
        else:
            if l2_regularization_const > 0.0:
                gradient_weights += model * (l2_regularization_const * batch_size)
            sys.stdout.write("\r                                                                \r")
            print "checking gradient..."
            finite_difference_model = Neural_Network_Weight(num_layers=model.num_layers)
            finite_difference_model.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
            
            direction = Neural_Network_Weight(num_layers=model.num_layers)
            direction.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
            epsilon = 1E-5
            for key in direction.bias.keys():
                print "at bias key", key
                for index in range(direction.bias[key].size):
                    direction.bias[key][0][index] = epsilon
                    #print direction.norm()
                    forward_loss = self.calculate_loss(batch_inputs, batch_labels, model = model + direction) #self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False, model = model + direction), batch_labels)
                    backward_loss = self.calculate_loss(batch_inputs, batch_labels, model = model - direction) #self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False, model = model - direction), batch_labels)
                    finite_difference_model.bias[key][0][index] = (forward_loss - backward_loss) / (2 * epsilon)
                    direction.bias[key][0][index] = 0.0
            for key in direction.weights.keys():
                print "at weight key", key
                for index0 in range(direction.weights[key].shape[0]):
                    for index1 in range(direction.weights[key].shape[1]):
                        direction.weights[key][index0][index1] = epsilon
                        forward_loss = self.calculate_loss(batch_inputs, batch_labels, model = model + direction) #self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False, model = model + direction), batch_labels)
                        backward_loss = self.calculate_loss(batch_inputs, batch_labels, model = model - direction) #self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False, model = model - direction), batch_labels)
                        finite_difference_model.weights[key][index0][index1] = (forward_loss - backward_loss) / (2 * epsilon)
                        direction.weights[key][index0][index1] = 0.0
            
            for layer_num in range(model.num_layers,0,-1):
                weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
                bias_cur_layer = str(layer_num)
                print "calculated gradient for bias", bias_cur_layer
                print gradient_weights.bias[bias_cur_layer]
                print "finite difference approximation for bias", bias_cur_layer
                print finite_difference_model.bias[bias_cur_layer]
                print "calculated gradient for weights", weight_cur_layer
                print gradient_weights.weights[weight_cur_layer]
                print "finite difference approximation for weights", weight_cur_layer
                print finite_difference_model.weights[weight_cur_layer]
            print "calculated gradient for bias 0"
            print gradient_weights.bias['0']
            print "finite difference approximation for bias 0"
            print finite_difference_model.bias['0']
            sys.exit()
        ##########################################################
    
    def calculate_fisher_diag_matrix(self, batch_inputs, batch_labels, check_gradient = False, model=None, l2_regularization_const = None): #compared with finite differences, seems kosher
        if model == None:
            model = self.model
        if l2_regularization_const == None:
            l2_regularization_const = self.l2_regularization_const
        
        if l2_regularization_const > 0.0:
            #the calculation is a bit more involved than gradient, will need derivative w.r.t. cross-entropy, and not loss, so use
            #calculate_gradient(), but set l2_regularization_const = 0.0
            average_grad_cross_entropy = self.calculate_gradient(batch_inputs, batch_labels, check_gradient=False, model=model, l2_regularization_const = 0.0)
        output_model = Neural_Network_Weight(num_layers=model.num_layers)
        output_model.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
        
        batch_size = batch_inputs.shape[0]
        hiddens = self.forward_first_order_methods(batch_inputs, model)
        
        weight_vec = hiddens[model.num_layers]
        
        for index in range(batch_size):
            weight_vec[index, batch_labels[index]] -= 1
        
        weight_cur_layer = ''.join([str(model.num_layers-1), str(model.num_layers)])
        bias_cur_layer = str(model.num_layers)
        output_model.bias[bias_cur_layer][0] = sum(weight_vec**2)
        
        output_model.weights[weight_cur_layer] = np.dot(np.transpose(hiddens[model.num_layers-1]**2), weight_vec**2)
        #propagate to sigmoid layers
        for layer_num in range(model.num_layers-1,0,-1):
            weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
            weight_next_layer = ''.join([str(layer_num),str(layer_num+1)])
            bias_cur_layer = str(layer_num)
            weight_vec = np.dot(weight_vec, np.transpose(model.weights[weight_next_layer])) * hiddens[layer_num] * (1-hiddens[layer_num]) #n_hid x n_out * (batchsize x n_out)
            output_model.bias[bias_cur_layer][0] = sum(weight_vec**2) #this is somewhat ugly
            output_model.weights[weight_cur_layer] = np.dot(np.transpose(hiddens[layer_num-1]**2), weight_vec**2)
        
        if not check_gradient:
            if l2_regularization_const > 0.0:
                return (output_model / batch_size + (model ** 2) * (l2_regularization_const ** 2) + 
                        (average_grad_cross_entropy * model) * (2 * l2_regularization_const))
            return output_model / batch_size
    
        ##### block for checking gradient only needed if you think that Fisher diagonal matrix has a problem ####################
        else:
            if l2_regularization_const > 0.0:
                output_model = (output_model + (model ** 2) * ((l2_regularization_const ** 2) * batch_size) + 
                                (average_grad_cross_entropy * model) * (2 * l2_regularization_const * batch_size))
                
            sys.stdout.write("\r                                                                \r")
            print "checking fisher diagonal matrix..."
            finite_difference_model = Neural_Network_Weight(num_layers=model.num_layers)
            finite_difference_model.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
            direction = Neural_Network_Weight(num_layers=model.num_layers)
            direction.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
            
            epsilon = 1E-5
            for key in direction.bias.keys():
                print "at bias key", key
                for index in range(direction.bias[key].size):
                    direction.bias[key][0][index] = epsilon
                    #print direction.norm()
                    for batch_idx in range(batch_size):
                        forward_loss = self.calculate_loss(batch_inputs[batch_idx], batch_labels[batch_idx], model = model + direction) #self.calculate_cross_entropy(self.forward_pass(batch_inputs[batch_idx], verbose=False, model = model + direction), batch_labels[batch_idx])
                        backward_loss = self.calculate_loss(batch_inputs[batch_idx], batch_labels[batch_idx], model = model - direction) #self.calculate_cross_entropy(self.forward_pass(batch_inputs[batch_idx], verbose=False, model = model - direction), batch_labels[batch_idx])
                        finite_difference_model.bias[key][0][index] += ((forward_loss - backward_loss) / (2 * epsilon)) ** 2
                    direction.bias[key][0][index] = 0.0
            for key in direction.weights.keys():
                print "at weight key", key
                for index0 in range(direction.weights[key].shape[0]):
                    for index1 in range(direction.weights[key].shape[1]):
                        direction.weights[key][index0][index1] = epsilon
                        for batch_idx in range(batch_size):
                            forward_loss = self.calculate_loss(batch_inputs[batch_idx], batch_labels[batch_idx], model = model + direction) #self.calculate_cross_entropy(self.forward_pass(batch_inputs[batch_idx], verbose=False, model = model + direction), batch_labels[batch_idx])
                            backward_loss = self.calculate_loss(batch_inputs[batch_idx], batch_labels[batch_idx], model = model - direction) #self.calculate_cross_entropy(self.forward_pass(batch_inputs[batch_idx], verbose=False, model = model - direction), batch_labels[batch_idx])
                            finite_difference_model.weights[key][index0][index1] += ((forward_loss - backward_loss) / (2 * epsilon)) ** 2
                        direction.weights[key][index0][index1] = 0.0
            
            for layer_num in range(model.num_layers,0,-1):
                weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
                bias_cur_layer = str(layer_num)
                print "calculated fisher diagonal for bias", bias_cur_layer
                print output_model.bias[bias_cur_layer]
                print "finite difference approximation for bias", bias_cur_layer
                print finite_difference_model.bias[bias_cur_layer]
                print "calculated fisher diagonal for weights", weight_cur_layer
                print output_model.weights[weight_cur_layer]
                print "finite difference approximation for weights", weight_cur_layer
                print finite_difference_model.weights[weight_cur_layer]
            sys.exit()
            ###########################################################################################
        
    def backward_pass(self, backward_inputs, hiddens, model=None): #need to test
        if model == None:
            model = self.model
        output_model = Neural_Network_Weight(num_layers=model.num_layers)
        output_model.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
        weight_vec = backward_inputs
        #average layers in batch
        weight_cur_layer = ''.join([str(model.num_layers-1), str(model.num_layers)])
        bias_cur_layer = str(model.num_layers)
        output_model.bias[bias_cur_layer][0] = sum(weight_vec)
        output_model.weights[weight_cur_layer] = np.dot(np.transpose(hiddens[model.num_layers-1]), weight_vec)
        #propagate to sigmoid layers
        for layer_num in range(model.num_layers-1,0,-1):
            weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
            weight_next_layer = ''.join([str(layer_num),str(layer_num+1)])
            bias_cur_layer = str(layer_num)
            weight_vec = np.dot(weight_vec, np.transpose(model.weights[weight_next_layer])) * hiddens[layer_num] * (1-hiddens[layer_num]) #n_hid x n_out * (batchsize x n_out)
            output_model.bias[bias_cur_layer][0] = sum(weight_vec) #this is somewhat ugly
            output_model.weights[weight_cur_layer] = np.dot(np.transpose(hiddens[layer_num-1]), weight_vec)
        return output_model
    def line_search(self, batch_inputs, batch_labels, direction, max_step_size=0.1, #completed, way expensive, should be compiled
                    max_line_searches=20, init_step_size=0.0, model = None, verbose=False): 
        """the line search algorithm is basically as follows
         we have directional derivative of p_k at cross_entropy(0), in gradient_direction, self.armijo_const, and self.wolfe_const, and stepsize_max, current cross-entropy in batch
         choose stepsize to be between 0 and stepsize_max (usually by finding minimum or quadratic, cubic, or quartic function)
        while loop
            evaluate cross-entropy at point weight + stepsize * gradient direction
            if numerical issue (cross_entropy is inf, etc).
                divide step_size by 2 and try again
            if fails first Wolfe condition (i.e., evaluated_cross_entropy > current_cross_entropy + self.armijo_const * stepsize * dir_deriv(cross_ent(0))
                interpolate between (prev_stepsize, cur_stepsize) and return that stepsize #we went too far in the current direction
            if not we made it past first Wolfe condition
            calculate directional derivative at proposed point
            if made it past second Wolfe condition (i.e., abs(prop_dir_deriv) <= -self.wolfe_const dir_deriv(0)
                finished line search
            elif dir_deriv(proposed) >= 0 #missed minimum before
                interp between current step size and previous one
             otherwise we essentially didn't go far enough with our step size, so find step_size between current_stepsize and max_stepsize"""
        
        
        if model == None:
            model = self.model
        #checks to see if armijo and wolfe constants are valid
        if self.armijo_const < 0 or self.armijo_const > 1:
            print "armijo constant (key armijo_const) must be between 0 and 1. Instead it is", self.armijo_const, "... Exiting now"
            sys.exit()
        if self.wolfe_const < 0 or self.wolfe_const > 1:
            print "wolfe constant (key wolfe_const) must be between 0 and 1. Instead it is", self.wolfe_const, "... Exiting now"
            sys.exit()
        if self.armijo_const >= self.wolfe_const:
            print "armijo constant (key armijo_const) but be less than wolfe constant (key wolfe_const). Instead armijo constant is", self.armijo_const,
            print "while the wolfe constant is", self.wolfe_const, "... Exiting now"
            sys.exit()
        excluded_keys = {'bias':['0'], 'weights':[]} #will have to change this later
        zero_step_loss = self.calculate_loss(batch_inputs, batch_labels, model) #self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False, model=model), batch_labels) #\phi_0
        zero_step_directional_derivative = direction.dot(self.calculate_gradient(batch_inputs, batch_labels, False, model), excluded_keys)
        if init_step_size == 0.0:
            step_size = max_step_size / 2
        else:
            step_size = init_step_size
        
        prev_step_size = 0
        prev_loss = zero_step_loss
        prev_directional_derivative = zero_step_directional_derivative
        (upper_bracket, upper_bracket_loss, upper_bracket_deriv, lower_bracket, lower_bracket_loss, lower_bracket_deriv) = [0 for _ in range(6)]
        
        for num_line_searches in range(1,max_line_searches+1): #looking for brackets
            if verbose: print "proposed step size is", step_size
            try:
                proposed_loss = self.calculate_loss(batch_inputs, batch_labels, model = model + direction * step_size) #self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False,
                                                                                                                        #model = model + direction * step_size), batch_labels)
            except FloatingPointError:
                print "encountered floating point error (likely under/overflow during forward pass), so decreasing step size by 1/2"
                step_size /= 2
                continue
            if math.isinf(proposed_loss) or math.isnan(proposed_loss): #numerical stability issues
                print "have numerical stability issues, so decreasing step size by 1/2"
                step_size /= 2
                continue
            if proposed_loss > zero_step_loss + self.armijo_const * step_size * zero_step_directional_derivative: #fails Armijo rule, but we have found our bracket
                # we now know that Wolfe conditions are satisfied between prev_step_size  and proposed_step_size
                if verbose: print "Armijo rule failed, so generating brackets"
                upper_bracket = step_size
                upper_bracket_loss = proposed_loss
                upper_bracket_deriv = direction.dot(self.calculate_gradient(batch_inputs, batch_labels, False, model), excluded_keys)
                lower_bracket = prev_step_size
                lower_bracket_loss = prev_loss
                lower_bracket_deriv = prev_directional_derivative
                break
            proposed_directional_derivative = direction.dot(self.calculate_gradient(batch_inputs, batch_labels, False, model = model + direction * step_size), excluded_keys)
            
            if abs(proposed_directional_derivative) <= -self.wolfe_const * zero_step_directional_derivative: #satisfies strong Wolfe condition
                if verbose:
                    print "Wolfe conditions satisfied"
                    print "returned step size", step_size
                return step_size
            elif proposed_directional_derivative >= 0:
                if verbose: print "went too far for second order condition, brackets found"
                lower_bracket = step_size
                lower_bracket_loss = proposed_loss
                lower_bracket_deriv = proposed_directional_derivative
                upper_bracket = prev_step_size
                upper_bracket_loss = prev_loss
                lower_bracket_deriv = prev_directional_derivative
                break
            else: #satisfies Armijo rule, but not 2nd Wolfe condition, so go out further
                if verbose: print "satisfies Armijo rule, but not 2nd Wolfe condition, so go out further"
                prev_step_size = step_size
                prev_loss = proposed_loss
                prev_directional_derivative = proposed_directional_derivative
                step_size = (prev_step_size + max_step_size) / 2
                
        #after first loop, weights are set prev_step_size * direction, with upper and lower brackets set, now find step size that
        #satisfy Wolfe conditions
        remaining_line_searches = max_line_searches - num_line_searches
        
        for _ in range(remaining_line_searches): #searching for good step sizes within bracket
            if verbose: print "upper bracket:", upper_bracket, "lower bracket:", lower_bracket
            step_size = self.interpolate_step_size((upper_bracket, upper_bracket_loss, upper_bracket_deriv),
                                                    (lower_bracket, lower_bracket_loss, lower_bracket_deriv))
            mix_const = np.abs((step_size - lower_bracket) / (upper_bracket - lower_bracket))
            if mix_const < 0.05 or mix_const > 0.95:
                step_size = (upper_bracket + lower_bracket) / 2
            if verbose: print "proposed step size is", step_size
            proposed_loss = self.calculate_loss(batch_inputs, batch_labels, model = model + direction * step_size) #self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False,model = model + direction * step_size), batch_labels)
            proposed_directional_derivative = direction.dot(self.calculate_gradient(batch_inputs, batch_labels, model = model + direction * step_size), excluded_keys)
            if proposed_loss > zero_step_loss + self.armijo_const * step_size * zero_step_directional_derivative or proposed_loss >= lower_bracket_loss:
                if verbose: print "Armijo rule failed, adjusting brackets"
                upper_bracket = step_size
                upper_bracket_loss = proposed_loss
                upper_bracket_deriv = proposed_directional_derivative
            else:
                if verbose: print "Armijo rule satisfied"
                if abs(proposed_directional_derivative) <= -self.wolfe_const * zero_step_directional_derivative: #satisfies strong Wolfe condition
                    if verbose:
                        print "satisfied Wolfe conditions"
                        print "returned step size", step_size
                    return step_size
                elif proposed_directional_derivative * (upper_bracket - lower_bracket) >= 0:
                    if verbose: print "went too far on step ... adjusting brackets"
                    upper_bracket = lower_bracket
                    upper_bracket_loss = lower_bracket_loss
                    upper_bracket_deriv = lower_bracket_deriv
                lower_bracket = step_size
                lower_bracket_loss = proposed_loss
                lower_bracket_deriv = proposed_directional_derivative
        if verbose: print "returning 0.0 step size"        
        return 0.0 #line search failed
    def interpolate_step_size(self, p1, p2): #completed
        """p1 and p2 are tuples in form (x,f(x), f'(x)) and spits out minimum step size based on cubic interpolation of data
            from page 56 of Numerical Optimization, Nocedal and Wright, 2nd edition"""
        if abs(p2[0] - p1[0]) < 1E-4:
            #print "difference between two step sizes is small. |p2 -p1| =", abs(p2[0] - p1[0]), "returning bisect"
            return (p1[0] + p2[0]) / 2
        b = (3 * (p2[1] - p1[1]) - (2 * p1[2] + p2[2])*(p2[0] - p1[0])) / (p2[0] - p1[0])**2
        a = (-2 * (p2[1] - p1[1]) + (p1[2] + p2[2])*(p2[0] - p1[0])) / (p2[0] - p1[0])**3
        if b**2 > 3 * a * p1[2]: #cubic interp
            return p1[0] + (-b + np.sqrt(b**2 - 3 * a * p1[2])) / (3 * a)
        elif b != 0: #quadratic interp
            return p1[0] - p1[2] / (2 * b)
        else: #bisect
            return (p1[0] + p2[0]) / 2
    def calculate_krylov_basis(self, batch_inputs, batch_labels, prev_direction, gradient = None, model = None, preconditioner = None): #need to test
        if model == None:
            model = self.model
        batch_size = batch_inputs.shape[0]
        krylov_basis = dict() #dictionary of weights, "directions" are weights
        excluded_keys = {'bias': ['0'], 'weights': []}
        krylov_basis['hessian'] = np.identity(self.krylov_num_directions+1)
        #will need to add preconditioning at some point
        hiddens = self.forward_first_order_methods(batch_inputs, model)
        if gradient == None:
            krylov_basis[0] = self.calculate_gradient(batch_inputs, batch_labels, False, model, hiddens=hiddens) #normed gradient for first direction
        else:
            krylov_basis[0] = gradient
        
        if preconditioner != None:
            krylov_basis[0] = krylov_basis[0] / preconditioner
        
        for basis_num in range(1,self.krylov_num_directions+1):
            sys.stdout.write("\r                                                                \r")
            sys.stdout.write("calculating basis number %d of %d" % (basis_num, self.krylov_num_directions)), sys.stdout.flush()
            #print "krylov basis norm is", krylov_basis[basis_num-1].norm(excluded_keys)
            krylov_basis[basis_num-1] /= krylov_basis[basis_num-1].norm(excluded_keys)
            #if basis_num < self.krylov_num_directions:
            second_order_direction = self.calculate_second_order_direction(batch_inputs, batch_labels, 
                                                                           direction = krylov_basis[basis_num-1], model = model, 
                                                                           second_order_type = self.second_order_matrix, hiddens = hiddens)
                #print "printing second_order_direction statistics"
                #second_order_direction.print_statistics()
                #will have to add preconditioning here

            if preconditioner != None and basis_num < self.krylov_num_directions:
                basis_direction = second_order_direction / preconditioner
            elif basis_num < self.krylov_num_directions:
                basis_direction = second_order_direction
            else:
                basis_direction = prev_direction
                
            for hessian_idx in range(basis_num):
                if self.krylov_use_hessian_preconditioner:
                    krylov_basis['hessian'][hessian_idx,basis_num-1] = second_order_direction.dot(krylov_basis[hessian_idx], excluded_keys)
                    krylov_basis['hessian'][basis_num-1,hessian_idx] = krylov_basis['hessian'][hessian_idx,basis_num-1]
                #orthogonalize direction
                basis_direction -= krylov_basis[hessian_idx] * basis_direction.dot(krylov_basis[hessian_idx], excluded_keys) #will be preconditioned here
            krylov_basis[basis_num] = basis_direction
        if self.krylov_use_hessian_preconditioner:
            second_order_direction = self.calculate_second_order_direction(batch_inputs, batch_labels, 
                                                                           direction = second_order_direction, model = model, 
                                                                           second_order_type = self.second_order_matrix, hiddens = hiddens)
            for hessian_idx in range(self.krylov_num_directions+1):
                krylov_basis['hessian'][hessian_idx,self.krylov_num_directions] = second_order_direction.dot(krylov_basis[hessian_idx], excluded_keys)
                krylov_basis['hessian'][self.krylov_num_directions,hessian_idx] = krylov_basis['hessian'][hessian_idx,self.krylov_num_directions]
        return krylov_basis
    def calculate_second_order_direction(self, inputs, labels, direction = None, model = None, second_order_type = None, hiddens = None, check_direction = False): #need to test
        #given an input direction direction, the function returns H*d, where H is the Hessian of the weight vector
        #the function does this efficient by using the Pearlmutter (1994) trick
        excluded_keys = {'bias': ['0'], 'weights': []}
        batch_size = inputs.shape[0]
        if model == None:
            model = self.model
        if direction == None:
            direction = self.calculate_gradient(inputs, labels, False, model)
        if second_order_type == None:
            second_order_type='gauss-newton' #other option is 'hessian'
        if hiddens == None:
            hiddens = self.forward_first_order_methods(inputs, model)    
        
        
        if second_order_type == 'gauss-newton':
            hidden_deriv = self.pearlmutter_forward_pass(labels, hiddens, model, direction, stop_at='output') #nbatch x nout
            second_order_direction = self.backward_pass(hidden_deriv[model.num_layers], hiddens, model)
        elif second_order_type == 'hessian':
            hidden_deriv = self.pearlmutter_forward_pass(labels, hiddens, model, direction, stop_at='output') #nbatch x nout
            second_order_direction = self.pearlmutter_backward_pass(hidden_deriv, labels, hiddens, model, direction)
        elif second_order_type == 'fisher':
            hidden_deriv = self.pearlmutter_forward_pass(labels, hiddens, model, direction, stop_at='loss') #nbatch x nout
            weight_vec = hiddens[model.num_layers] 
            for index in range(batch_size):
                weight_vec[index, labels[index]] -= 1 
            weight_vec *= hidden_deriv[model.num_layers+1][:, np.newaxis]
            second_order_direction = self.backward_pass(weight_vec, hiddens, model) 
        else:
            print second_order_type, "is not a valid type. Acceptable types are gauss-newton, hessian, and fisher... Exiting now..."
            sys.exit()
            
        if not check_direction:
            if self.l2_regularization_const > 0.0:
                return second_order_direction / batch_size + direction * self.l2_regularization_const
            return second_order_direction / batch_size
        
        ##### check direction only if you think there is a problem #######
        else:
            finite_difference_model = Neural_Network_Weight(num_layers=model.num_layers)
            finite_difference_model.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
            epsilon = 1E-5
            
            if second_order_type == 'gauss-newton':
                #assume that pearlmutter forward pass is correct because the function has a check_gradient flag to see if it's is
                sys.stdout.write("\r                                                                \r")
                sys.stdout.write("checking Gv\n"), sys.stdout.flush()
                weight_cur_layer = ''.join([str(model.num_layers-1),str(model.num_layers)])
                bias_cur_layer = str(model.num_layers)
                linear_out = self.forward_pass_linear(inputs, verbose=False, model = model)
                finite_diff_forward = self.forward_pass_linear(inputs, verbose=False, model = model + direction * epsilon)
                finite_diff_backward = self.forward_pass_linear(inputs, verbose=False, model = model - direction * epsilon)
                finite_diff_jacobian_vec = (finite_diff_forward - finite_diff_backward) / (2 * epsilon)
                finite_diff_HJv = np.zeros(finite_diff_jacobian_vec.shape)
                
                num_outputs = len(linear_out[0])
                collapsed_hessian = np.zeros((num_outputs,num_outputs))
                for batch_index in range(batch_size):
                    #calculate collapsed Hessian
                    direction1 = np.zeros(finite_diff_forward[0].shape)
                    direction2 = np.zeros(finite_diff_forward[0].shape)
                    for index1 in range(len(finite_diff_forward[0])):
                        for index2 in range(len(finite_diff_forward[0])):
                            direction1[index1] = epsilon
                            direction2[index2] = epsilon
                            loss_plus_plus = self.calculate_cross_entropy(self.softmax(np.array([linear_out[batch_index] + direction1 + direction2])), labels[batch_index:batch_index+1])
                            loss_plus_minus = self.calculate_cross_entropy(self.softmax(np.array([linear_out[batch_index] + direction1 - direction2])), labels[batch_index:batch_index+1])
                            loss_minus_plus = self.calculate_cross_entropy(self.softmax(np.array([linear_out[batch_index] - direction1 + direction2])), labels[batch_index:batch_index+1])
                            loss_minus_minus = self.calculate_cross_entropy(self.softmax(np.array([linear_out[batch_index] - direction1 - direction2])), labels[batch_index:batch_index+1])
                            collapsed_hessian[index1,index2] = (loss_plus_plus + loss_minus_minus - loss_minus_plus - loss_plus_minus) / (4 * epsilon * epsilon)
                            direction1[index1] = 0.0
                            direction2[index2] = 0.0
                    print collapsed_hessian
                    out = self.softmax(linear_out[batch_index:batch_index+1])
                    print np.diag(out[0]) - np.outer(out[0], out[0])
                    finite_diff_HJv[batch_index] += np.dot(collapsed_hessian, finite_diff_jacobian_vec[batch_index])
                    
                for batch_index in range(batch_size):
                    #calculate J'd = J'HJv
                    update = Neural_Network_Weight(num_layers=model.num_layers)
                    update.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
                    for key in direction.bias.keys():
                        print "at bias key", key
                        for index in range(direction.bias[key].size):
                            update.bias[key][0][index] = epsilon
                            #print direction.norm()
                            forward_loss = self.forward_pass_linear(inputs[batch_index:batch_index+1], verbose=False, model = model + update)
                            backward_loss = self.forward_pass_linear(inputs[batch_index:batch_index+1], verbose=False, model = model - update)
                            finite_difference_model.bias[key][0][index] += np.dot((forward_loss - backward_loss) / (2 * epsilon), finite_diff_HJv[batch_index])
                            update.bias[key][0][index] = 0.0
                    for key in direction.weights.keys():
                        print "at weight key", key
                        for index0 in range(direction.weights[key].shape[0]):
                            for index1 in range(direction.weights[key].shape[1]):
                                update.weights[key][index0][index1] = epsilon
                                forward_loss = self.forward_pass_linear(inputs[batch_index:batch_index+1], verbose=False, model= model + update)
                                backward_loss = self.forward_pass_linear(inputs[batch_index:batch_index+1], verbose=False, model= model - update)
                                finite_difference_model.weights[key][index0][index1] += np.dot((forward_loss - backward_loss) / (2 * epsilon), finite_diff_HJv[batch_index])
                                update.weights[key][index0][index1] = 0.0
            elif second_order_type == 'hessian':
                sys.stdout.write("\r                                                                \r")
                sys.stdout.write("checking Hv\n"), sys.stdout.flush()
                for batch_index in range(batch_size):
                    #assume that gradient calculation is correct
                    print "at batch index", batch_index
                    update = Neural_Network_Weight(num_layers=model.num_layers)
                    update.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
                    
                    current_gradient = self.calculate_gradient(inputs[batch_index:batch_index+1], labels[batch_index:batch_index+1], False, model, l2_regularization_const = 0.)
                    
                    for key in finite_difference_model.bias.keys():
                        for index in range(direction.bias[key].size):
                            update.bias[key][0][index] = epsilon
                            forward_loss = self.calculate_gradient(inputs[batch_index:batch_index+1], labels[batch_index:batch_index+1], False, 
                                                                   model = model + update, l2_regularization_const = 0.)
                            backward_loss = self.calculate_gradient(inputs[batch_index:batch_index+1], labels[batch_index:batch_index+1], False, 
                                                                    model = model - update, l2_regularization_const = 0.)
                            finite_difference_model.bias[key][0][index] += direction.dot((forward_loss - backward_loss) / (2 * epsilon), excluded_keys)
                            update.bias[key][0][index] = 0.0
        
                    for key in finite_difference_model.weights.keys():
                        for index0 in range(direction.weights[key].shape[0]):
                            for index1 in range(direction.weights[key].shape[1]):
                                update.weights[key][index0][index1] = epsilon
                                forward_loss = self.calculate_gradient(inputs[batch_index:batch_index+1], labels[batch_index:batch_index+1], False, 
                                                                       model = model + update, l2_regularization_const = 0.) 
                                backward_loss = self.calculate_gradient(inputs[batch_index:batch_index+1], labels[batch_index:batch_index+1], False, 
                                                                        model = model - update, l2_regularization_const = 0.)
                                finite_difference_model.weights[key][index0][index1] += direction.dot((forward_loss - backward_loss) / (2 * epsilon), excluded_keys)
                                update.weights[key][index0][index1] = 0.0
            elif second_order_type == 'fisher':
                sys.stdout.write("\r                                                                \r")
                sys.stdout.write("checking Fv\n"), sys.stdout.flush()
                for batch_index in range(batch_size):
                    #assume that gradient calculation is correct
                    print "at batch index", batch_index
                    current_gradient = self.calculate_gradient(inputs[batch_index:batch_index+1], labels[batch_index:batch_index+1], False, model, l2_regularization_const = 0.)                
                    finite_difference_model += current_gradient * current_gradient.dot(direction, excluded_keys)
                    
            for layer_num in range(model.num_layers,0,-1):
                weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
                bias_cur_layer = str(layer_num)
                print "calculated second order direction for bias", bias_cur_layer
                print second_order_direction.bias[bias_cur_layer]
                print "finite difference approximation for bias", bias_cur_layer
                print finite_difference_model.bias[bias_cur_layer]
                print "calculated second order direction for weights", weight_cur_layer
                print second_order_direction.weights[weight_cur_layer]
                print "finite difference approximation for weights", weight_cur_layer
                print finite_difference_model.weights[weight_cur_layer]
            sys.exit()
        ##########################################################
                                
    def calculate_classification_statistics(self, features, labels, model=None):
        if model == None:
            model = self.model
        
        excluded_keys = {'bias': ['0'], 'weights': []}
        
        if self.do_backprop == False:
            classification_batch_size = 4096
        else:
            classification_batch_size = max(self.backprop_batch_size, 4096)
        
        batch_index = 0
        end_index = 0
        cross_entropy = 0.0
        num_correct = 0
        num_examples = features.shape[0]
        while end_index < num_examples: #run through the batches
            end_index = min(batch_index+classification_batch_size, num_examples)
            output = self.forward_pass(features[batch_index:end_index], verbose=False, model=model)
            cross_entropy += self.calculate_cross_entropy(output, labels[batch_index:end_index])
            
            #don't use calculate_classification_accuracy() because of possible rounding error
            prediction = output.argmax(axis=1).reshape(labels[batch_index:end_index].shape)
            num_correct += np.sum(prediction == labels[batch_index:end_index])
            batch_index += classification_batch_size
        
        loss = cross_entropy
        if self.l2_regularization_const > 0.0:
            loss += (model.norm(excluded_keys) ** 2) * self.l2_regularization_const
        return cross_entropy, num_correct, num_examples, loss
    def pearlmutter_forward_pass(self, labels, hiddens, model, direction, check_gradient=False, stop_at='output'): #need to test
        """let f be a function from inputs to outputs
        consider the weights to be a vector w of parameters to be optimized, (and direction d to be the same)
        pearlmutter_forward_pass calculates d' \jacobian_w f
        hiddens[0] are the inputs
        stop_at is either 'linear', 'output', or 'loss' """
        
        batch_size = len(labels)
        hidden_deriv = dict()
        hidden_deriv[1] = self.weight_matrix_multiply(hiddens[0], direction.weights['01'], direction.bias['1']) * hiddens[1] * (1-hiddens[1])
        for layer_num in range(1,model.num_layers-1):
            weight_cur_layer = ''.join([str(layer_num), str(layer_num+1)])
            bias_cur_layer = str(layer_num+1)
            hidden_deriv[layer_num+1] = ((self.weight_matrix_multiply(hiddens[layer_num], direction.weights[weight_cur_layer], 
                                                                      direction.bias[bias_cur_layer]) +
                                          np.dot(hidden_deriv[layer_num], model.weights[weight_cur_layer])) *
                                          hiddens[layer_num+1] * (1-hiddens[layer_num+1]) )
        #update last layer, assuming logistic regression
        
        weight_cur_layer = ''.join([str(model.num_layers-1), str(model.num_layers)])
        bias_cur_layer = str(model.num_layers)
        linear_layer = (self.weight_matrix_multiply(hiddens[model.num_layers-1], direction.weights[weight_cur_layer], 
                                                    direction.bias[bias_cur_layer]) +
                        np.dot(hidden_deriv[model.num_layers-1], model.weights[weight_cur_layer]))
        if stop_at == 'linear':
            hidden_deriv[model.num_layers] = linear_layer
        else:
            hidden_deriv[model.num_layers] = linear_layer * hiddens[model.num_layers] - hiddens[model.num_layers] * np.sum(linear_layer * hiddens[model.num_layers], axis=1)[:,np.newaxis]
        if stop_at == 'loss':
            hidden_deriv[model.num_layers+1] = -np.array([(hidden_deriv[model.num_layers][index, labels[index]] / hiddens[model.num_layers][index, labels[index]])[0] for index in range(batch_size)])
            
        if not check_gradient:
            return hidden_deriv
        #compare with finite differences approximation
        else:
            epsilon = 1E-10
            if stop_at == 'linear':
                calculated = hidden_deriv[model.num_layers]
                finite_diff_forward = self.forward_pass_linear(hiddens[0], verbose=False, model = model + direction * epsilon)
                finite_diff_backward = self.forward_pass_linear(hiddens[0], verbose=False, model = model - direction * epsilon)
            elif stop_at == 'output':
                calculated = hidden_deriv[model.num_layers]
                finite_diff_forward = self.forward_pass(hiddens[0], verbose=False, model = model + direction * epsilon)
                finite_diff_backward = self.forward_pass(hiddens[0], verbose=False, model = model - direction * epsilon)
            elif stop_at == 'loss':
                calculated = hidden_deriv[model.num_layers + 1]
                finite_diff_forward = -np.log([max(self.forward_pass(hiddens[0], verbose=False, model = model + direction * epsilon).item((x,labels[x])),1E-12) for x in range(labels.size)]) 
                finite_diff_backward =  -np.log([max(self.forward_pass(hiddens[0], verbose=False, model = model - direction * epsilon).item((x,labels[x])),1E-12) for x in range(labels.size)]) 
            print "pearlmutter calculation"
            print calculated
            print "finite differences approximation, epsilon", epsilon
            print ((finite_diff_forward - finite_diff_backward) / (2 * epsilon))
            sys.exit()
        ######################################################################
        
    def pearlmutter_backward_pass(self, hidden_deriv, batch_labels, hiddens, model, direction):
        if model == None:
            model = self.model
        output_model = Neural_Network_Weight(num_layers=model.num_layers)
        output_model.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
        hidden_deriv[0] = np.zeros(hiddens[0].shape)
        #derivative of log(cross-entropy softmax)
        weight_vec = hiddens[model.num_layers] #batchsize x n_outputs
        batch_size = len(batch_labels)

        for index in range(batch_size):
            weight_vec[index, batch_labels[index]] -= 1
        weight_vec_deriv = hidden_deriv[model.num_layers]
        #average layers in batch
        weight_cur_layer = ''.join([str(model.num_layers-1), str(model.num_layers)])
        bias_cur_layer = str(model.num_layers)
        output_model.bias[bias_cur_layer][0] = sum(weight_vec_deriv)
        output_model.weights[weight_cur_layer] = (np.dot(np.transpose(hiddens[model.num_layers-1]), weight_vec_deriv) +
                                                  np.dot(np.transpose(hidden_deriv[model.num_layers-1]), weight_vec))
        #propagate to sigmoid layers
        for layer_num in range(model.num_layers-1,0,-1):
            weight_cur_layer = ''.join([str(layer_num-1),str(layer_num)])
            weight_next_layer = ''.join([str(layer_num),str(layer_num+1)])
            bias_cur_layer = str(layer_num)
            d_hidden = hiddens[layer_num] * (1-hiddens[layer_num]) #derivative of sigmoid
            d2_hidden_div_d_hidden = (1. - 2. * hiddens[layer_num])
            old_weight_vec = weight_vec
            weight_vec = np.dot(weight_vec, np.transpose(model.weights[weight_next_layer])) * d_hidden
            weight_vec_deriv = (weight_vec * d2_hidden_div_d_hidden * hidden_deriv[layer_num] + 
                                (np.dot(old_weight_vec, np.transpose(direction.weights[weight_next_layer])) +
                                 np.dot(weight_vec_deriv, np.transpose(model.weights[weight_next_layer]))) * d_hidden)
            output_model.bias[bias_cur_layer][0] = sum(weight_vec_deriv) #this is somewhat ugly
            output_model.weights[weight_cur_layer] = (np.dot(np.transpose(hiddens[layer_num-1]), weight_vec_deriv) +
                                                      np.dot(np.transpose(hidden_deriv[layer_num-1]), weight_vec))
        
        return output_model
    def calculate_subspace_cross_entropy(self, parameters, basis, inputs, labels, model = None):
        #helper function for scipy bfgs
        if model == None:
            model = self.model
        num_directions = len(parameters)
        
        model_update = basis[0] * parameters[0]
        for dim in range(1, num_directions):
            model_update += basis[dim] * parameters[dim]
        
        return self.calculate_loss(inputs, labels, model = model + model_update) #self.calculate_cross_entropy(self.forward_pass(inputs, verbose = False, model = model + model_update), labels)
    def calculate_subspace_gradient(self, parameters, basis, inputs, labels, model = None):
        #helper function for scipy bfgs
        if model == None:
            model = self.model
        excluded_keys = {'bias': ['0'], 'weights': []}
        subspace_gradient = np.zeros(parameters.shape)
        num_directions = len(parameters)
        batch_size = inputs.shape[0]
        
        model_update = basis[0] * parameters[0]
        for dim in range(1, num_directions):
            model_update += basis[dim] * parameters[dim]
            
        model_gradient = self.calculate_gradient(inputs, labels, False, model = model + model_update)
        
        for dim in range(num_directions):
            subspace_gradient[dim] = model_gradient.dot(basis[dim], excluded_keys)
        return subspace_gradient
    def bfgs(self, batch_inputs, batch_labels, basis, num_epochs, model = None, verbose = False): #compared with scipy implementation, looks great, need to add gradient termination condition
        #helper function for Krylov subspace methods
        #given a basis of n directions B, bfgs attempts
        #to find the optimal step size a (which is an element of R^n)
        #such that w_0 + B*a gives the lowest error
        if model == None:
            model = self.model
        if verbose:
            print "calculating classification statistics before BFGS step"
            classification_stats = self.calculate_classification_statistics(batch_inputs, batch_labels, model = model)
            print "cross-entropy before BFGS is", classification_stats[0]
            print "number correctly classified is", classification_stats[1], "of", classification_stats[2]
        
        num_directions = self.krylov_num_directions + 1
        identity_mat = np.identity(num_directions)
            
        cur_step = np.zeros(num_directions)
        subspace_prev_gradient = np.zeros(num_directions)
        #finite_diff_approx = copy.deepcopy(cur_step)
        bfgs_mat = np.identity(num_directions)
        
        init_step_size = 1.0
        subspace_cur_gradient = self.calculate_subspace_gradient(np.zeros(num_directions), basis, batch_inputs, batch_labels, model)
        #for dim in range(num_directions):
        #    subspace_cur_gradient[dim] = model_gradient.dot(basis[dim], excluded_keys)
            #finite_diff_approx[dim] = ((self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False, model=model + basis[dim] * 0.0005), batch_labels) 
            #                            - (self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False, model=model), batch_labels)) )
            #                           / 0.0005 / batch_size)
        #print "cur_gradient is\n", cur_gradient
        #print "finite diff approximation is\n", finite_diff_approx
        model_update = Neural_Network_Weight(model.num_layers)
        model_update.init_zero_weights(model.get_architecture(), last_layer_logistic=True)
        #print finite_diff_approx / cur_gradient
        
        
        for epoch in range(num_epochs):
            print "\r                                                                \r", #clear line
            sys.stdout.write("\rbfgs epoch %d of %d\r" % (epoch+1, num_epochs)), sys.stdout.flush()
            #print "norm of the subspace gradient is", np.linalg.norm(subspace_cur_gradient)
            #print "bfgs matrix is"
            #print bfgs_mat
            #print "current gradient is"
            #print cur_gradient
            subspace_direction = -np.dot(bfgs_mat, subspace_cur_gradient)
            #print "provisional mix is", provisional_mix
            model_direction = basis[0] * subspace_direction[0]
            for dim in range(1, num_directions):
                model_direction += basis[dim] * subspace_direction[dim]
            #dir_derivative = gradient.dot(direction, excluded_keys)
            #print ", before line search, directional derivative is", dir_derivative
            step = self.line_search(batch_inputs, batch_labels, model_direction, max_step_size=1.5, 
                                    max_line_searches=self.num_line_searches, init_step_size=init_step_size, 
                                    model = model + model_update, verbose=False)
            if step == 0.0:
                print "\rline search failed, returning current step\r", #not updating any parameters"
                return cur_step
            else:
                prev_step = copy.deepcopy(cur_step)
                cur_step += subspace_direction * step
            model_update = basis[0] * cur_step[0]
            for dim in range(1,self.krylov_num_directions+1):
                model_update += basis[dim] * cur_step[dim]
            
            subspace_prev_gradient = copy.deepcopy(subspace_cur_gradient)
            subspace_cur_gradient = self.calculate_subspace_gradient(cur_step, basis, batch_inputs, batch_labels, model)
            if verbose:
                print "calculating classification statistics after BFGS step"
                classification_stats = self.calculate_classification_statistics(batch_inputs, batch_labels, model = model + model_update)
                print "cross-entropy after BFGS epoch", epoch, "is", classification_stats[0]
                print "number correctly classified is", classification_stats[1], "of", classification_stats[2]
            
            
            step_condition = cur_step - prev_step
            grad_condition = subspace_cur_gradient - subspace_prev_gradient
            curvature_condition = np.dot(step_condition, grad_condition)
            #print "curvature condition is", curvature_condition
            bfgs_mat = (np.dot((identity_mat - np.outer(step_condition, grad_condition) / curvature_condition), 
                                  np.dot(bfgs_mat, (identity_mat - np.outer(grad_condition, step_condition) / curvature_condition)))
                                  + np.outer(step_condition, step_condition) / curvature_condition)
            U,s,V = np.linalg.svd(bfgs_mat)
            #print "singular values of bfgs matrix are", s
            condition_number = max(s) / min(s)
            if condition_number > 30000.0:
                print "condition number of bfgs matrix is too high:", condition_number, "so returning current step\r"
                return cur_step
        return cur_step
                 
    def conjugate_gradient(self, batch_inputs, batch_labels, num_epochs, model = None, damping_factor = 0.0, #seems to be correct, compare with conjugate_gradient.py
                           verbose = False, preconditioner = None, gradient = None, second_order_type='gauss-newton', 
                           init_search_direction = None):
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
        
        model_update = Neural_Network_Weight(num_layers=model.num_layers)
        model_update.init_zero_weights(self.model.get_architecture(), last_layer_logistic=True, verbose=False)
        
        batch_size = batch_inputs.shape[0]
        hiddens = self.forward_first_order_methods(batch_inputs, model)
        if gradient == None:
            gradient = self.calculate_gradient(batch_inputs, batch_labels, False, model = model)
        
        if init_search_direction == None:
            model_vals.append(0)
            residual = gradient 
        else:
            second_order_direction = self.calculate_second_order_direction(batch_inputs, batch_labels, init_search_direction, 
                                                                           model, second_order_type=second_order_type, hiddens = hiddens)
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
                second_order_direction = self.calculate_second_order_direction(batch_inputs, batch_labels, search_direction, model, second_order_type=second_order_type, hiddens = hiddens) + search_direction * damping_factor
            else:
                second_order_direction = self.calculate_second_order_direction(batch_inputs, batch_labels, search_direction, model, second_order_type=second_order_type, hiddens = hiddens)
                                                                            
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

def init_arg_parser():
    required_variables = dict()
    all_variables = dict()
    required_variables['train'] = [ 'feature_file_name', 'output_name']
    all_variables['train'] = required_variables['train'] + ['label_file_name', 'hiddens_structure', 'weight_matrix_name', 
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
    required_variables['test'] =  ['feature_file_name', 'weight_matrix_name', 'output_name']
    all_variables['test'] =  required_variables['test'] + ['label_file_name']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='mode for DNN, either train or test', required=False)
    parser.add_argument('--config_file', help='configuration file to read in you do not want to input arguments via command line', required=False)
    for argument in all_variables['train']:
        parser.add_argument('--' + argument, required=False)
    for argument in all_variables['test']:
        if argument not in all_variables['train']:
            parser.add_argument('--' + argument, required=False)
    return parser

if __name__ == '__main__':
    #script_name, config_filename = sys.argv
    #print "Opening config file: %s" % config_filename
    script_name = sys.argv[0]
    parser = init_arg_parser()
    config_dictionary = vars(parser.parse_args())
    
    if config_dictionary['config_file'] != None :
        config_filename = config_dictionary['config_file']
        print "Since", config_filename, "is specified, ignoring other arguments"
        try:
            config_file=open(config_filename)
        except IOError:
            print "Could open file", config_filename, ". Usage is ", script_name, "<config file>... Exiting Now"
            sys.exit()
        
        del config_dictionary
        
        #read lines into a configuration dictionary, skipping lines that begin with #
        config_dictionary = dict([line.replace(" ", "").strip(' \n\t').split('=') for line in config_file 
                                  if not line.replace(" ", "").strip(' \n\t').startswith('#') and '=' in line])
        config_file.close()
    else:
        #remove empty keys
        config_dictionary = dict([(arg,value) for arg,value in config_dictionary.items() if value != None])

    try:
        mode=config_dictionary['mode']
    except KeyError:
        print 'No mode found, must be train or test... Exiting now'
        sys.exit()
    else:
        if (mode != 'train') and (mode != 'test'):
            print "Mode", mode, "not understood. Should be either train or test... Exiting now"
            sys.exit()
    
    if mode == 'test':
        test_object = NN_Tester(config_dictionary)
    else: #mode ='train'
        train_object = NN_Trainer(config_dictionary)
        train_object.train()
        
    print "Finished without Runtime Error!" 
                
                