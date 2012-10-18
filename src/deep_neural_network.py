'''

Created on Jul 20, 2012

@author: sumanravuri

Deep Neural Network with arbitrary number of hidden layers with these features:
* pretraining
* backprop with either stochastic or conjugate gradient descent, or 2nd order methods
* L2 regularization
* Early Stopping
* CUDA speedup

Oct 11, 2012 fixed error in backprop_steepest_descent
'''

import sys
import numpy
import scipy.io as sp
import math

class Neural_Network(object):
    #features are stored in format ndata x nvis
    #weights are stored as nvis x nhid at feature level
    #biases are stored as 1 x nhid
    #rbm_type is either gaussian_bernoulli, bernoulli_bernoulli, notrbm_logistic
    def __init__(self, config_dictionary): #completed
        #variables for Neural Network: feature_file_name(read from)
        self.feature_file_name = self.default_variable_define(config_dictionary, 'feature_file_name', arg_type='string')
        self.features = self.read_feature_file()
        self.output_name = self.default_variable_define(config_dictionary, 'output_name', arg_type='string')
        
        self.required_variables = {}
        self.all_variables = {}
        self.required_variables['train'] = ['mode', 'feature_file_name', 'output_name']
        self.all_variables['train'] = self.required_variables['train'] + ['label_file_name', 'architecture', 'weight_matrix_name', 
                               'initial_weight_max', 'initial_weight_min', 'initial_bias_max', 'initial_bias_min', 
                               'do_pretrain', 'pretrain_method', 'pretrain_iterations', 
                               'pretrain_learning_rate', 'pretrain_batch_size',
                               'do_backprop', 'backprop_method', 'backprop_batch_size', 'steepest_learning_rate',
                               'conjugate_num_epochs', 'conjugate_num_line_searches', 'conjugate_max_iterations', 'conjugate_const_type']
        self.required_variables['test'] =  ['mode', 'feature_file_name', 'weight_matrix_name', 'output_name']
        self.all_variables['test'] =  self.required_variables['test'] + ['label_file_name']
    def dump_config_vals(self):
        print "********************************************************************************"
        print "Neural Network configuration is as follows:"
        
        for key in self.all_variables[self.mode]:
            if hasattr(self,key):
                print key, "=", eval('self.' + key)
            else:
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
        dist = {}
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
                    string_distances = numpy.array([self.levenshtein_string_edit_distance(var, string2) for string2 in self.all_variables[correct_mode]])
                    print "perhaps you meant ***", self.all_variables[correct_mode][numpy.argmin(string_distances)], "\b*** (levenshtein string edit distance", numpy.min(string_distances), "\b) instead of ***", var, "\b***?"
                if exit_flag == False:
                    print "Because of above error, will exit after checking rest of keys"
                    exit_flag = True
        
        if exit_flag:
            print "Exiting now"
            sys.exit()
        else:
            print "seems copacetic"
    def check_labels(self): #want to prune non-contiguous labels
        print "Checking labels..."
        if len(self.labels.shape) != 1 and ((len(self.labels.shape) == 2 and self.labels.shape[1] != 1) or len(self.labels.shape) > 2):
            print "labels need to be in (n_samples) or (n_samples,1) format and the shape of labels is ", self.labels.shape, "... Exiting now"
            sys.exit()
        if self.labels.size != self.features.shape[0]:
            print "Number of examples in feature file: ", self.features.shape[0], " does not equal size of label file, ", self.labels.size, "... Exiting now"
            sys.exit()
        if  [i for i in numpy.unique(self.labels)] != range(numpy.max(self.labels)+1):
            print "Labels need to be in the form 0,1,2,....,n,... Exiting now"
            sys.exit()
        label_counts = numpy.bincount(numpy.ravel(self.labels)) #[self.labels.count(x) for x in range(numpy.max(self.labels)+1)]
        print "distribution of labels is:"
        for x in range(len(label_counts)):
            print "#", x, "\b's:", label_counts[x]            
        print "labels seem copacetic"
    def check_weights(self): #completed
        #checks weights to see if following conditions are true
        # *feature dimension equal to number of rows of first layer (if weights are stored in n_rows x n_cols)
        # *n_cols of (n-1)th layer == n_rows of nth layer
        # if only one layer, that weight layer type is logistic, gaussian_bernoulli or bernoulli_bernoulli
        # check is biases match weight values
        # if multiple layers, 0 to (n-1)th layer is gaussian bernoulli RBM or bernoulli bernoulli RBM and last layer is logistic regression
        
        #if below is true, not running in logistic regression mode, so first layer must be an RBM
        print "Checking weights...",
        if self.num_layers > 1: 
            if (self.weights['weights01_type'] != 'gaussian_bernoulli' and
                self.weights['weights01_type'] != 'bernoulli_bernoulli'):
                print self.weights['weights01_type'], "is not valid RBM type. Must be either gaussian_bernoulli or bernoulli_bernoulli... Exiting now"
                sys.exit()
        
        #check if feature dimension = n_vis of first layer
        if self.features.shape[1] != self.weights['weights01'].shape[0]: 
            print ("Number of feature dimensions: ", self.features.shape[1], 
                   " does not equal input dimensions", self.weights['weights01'].shape[0])
            sys.exit()
        
        #check biases
        if self.weights['bias0'].shape[1] != self.weights['weights01'].shape[0]:
            print ("Number of visible bias dimensions: ", self.weights['bias0'].shape[1],
                   " of layer 0 does not equal visible weight dimensions ", self.weights['weights01'].shape[0])
            sys.exit()
        if self.weights['bias1'].shape[1] != self.weights['weights01'].shape[1]:
            print ("Number of hidden bias dimensions: ", self.weights['bias1'].shape[1],
                   " of layer 0 does not equal hidden weight dimensions ", self.weights['weights01'].shape[1])
            sys.exit()
        
        #intermediate layers need to have correct shape and RBM type
        for idx in range(1,self.num_layers-1): 
            weight_prev_layer = ''.join(['weights',str(idx-1),str(idx)])
            weight_cur_layer = ''.join(['weights',str(idx),str(idx+1)])
            weight_cur_layer_type = ''.join([weight_cur_layer, '_type'])
            bias_prev_layer = ''.join(['bias',str(idx)])
            bias_cur_layer = ''.join(['bias',str(idx+1)])
            #check shape
            if self.weights[weight_prev_layer].shape[1] != self.weights[weight_cur_layer].shape[0]:
                print "Dimensionality of", weight_prev_layer, "\b:", self.weights[weight_prev_layer].shape, "does not match dimensionality of", weight_cur_layer, "\b:",self.weights[weight_cur_layer].shape
                print "The second dimension of", weight_prev_layer, "must equal the first dimension of", weight_cur_layer
                sys.exit()
            #check RBM type
            if (self.weights[weight_cur_layer_type] != 'gaussian_bernoulli' and
            self.weights[weight_cur_layer_type] != 'bernoulli_bernoulli'):
                print self.weights[weight_cur_layer_type], "is not valid RBM type. Must be either gaussian_bernoulli or bernoulli_bernoulli"
                sys.exit()
            #check biases
            if self.weights[bias_prev_layer].shape[1] != self.weights[weight_cur_layer].shape[0]:
                print "Number of visible bias dimensions:", self.weights[bias_prev_layer].shape[1], "of layer", weight_cur_layer, "does not equal visible weight dimensions:", self.weights[weight_cur_layer].shape[0]
                sys.exit()
            if self.weights[bias_cur_layer].shape[1] != self.weights[weight_cur_layer].shape[1]:
                print "Number of hidden bias dimensions:", self.weights[bias_cur_layer].shape[1],"of layer", weight_cur_layer, "does not equal hidden weight dimensions", self.weights[weight_cur_layer].shape[1]
                sys.exit()
        
        #check last layer
        idx = self.num_layers-1
        weight_prev_layer = ''.join(['weights',str(idx-1),str(idx)])
        weight_cur_layer = ''.join(['weights',str(idx),str(idx+1)])
        weight_cur_layer_type = ''.join([weight_cur_layer, '_type'])
        bias_prev_layer = ''.join(['bias',str(idx)])
        bias_cur_layer = ''.join(['bias',str(idx+1)])
        #check if last layer is of type notrbm_logistic
        if (self.weights[weight_cur_layer_type] != 'notrbm_logistic' and
            self.weights[weight_cur_layer_type] != 'gaussian_bernoulli' and
            self.weights[weight_cur_layer_type] != 'bernoulli_bernoulli'):
            print self.weights[weight_cur_layer_type], " is not valid type for last layer. It must be gaussian_bernoulli, bernoulli_bernoulli, notrbm_logistic"
            sys.exit()
        #check shape if hidden layer is used
        if self.num_layers > 1:
            if self.weights[weight_prev_layer].shape[1] != self.weights[weight_cur_layer].shape[0]:
                print "Dimensionality of", weight_prev_layer, "\b:", self.weights[weight_prev_layer].shape, "does not match dimensionality of", weight_cur_layer, "\b:",self.weights[weight_cur_layer].shape
                print "The second dimension of", weight_prev_layer, "must equal the first dimension of", weight_cur_layer
                sys.exit()
            #check biases
            if self.weights[bias_prev_layer].shape[1] != self.weights[weight_cur_layer].shape[0]:
                print "Number of visible bias dimensions:", self.weights[bias_prev_layer].shape[1], "of layer", weight_cur_layer, "does not equal visible weight dimensions:", self.weights[weight_cur_layer].shape[0]
                sys.exit()
            if self.weights[bias_cur_layer].shape[1] != self.weights[weight_cur_layer].shape[1]:
                print "Number of hidden bias dimensions:", self.weights[bias_cur_layer].shape[1],"of layer", weight_cur_layer, "does not equal hidden weight dimensions", self.weights[weight_cur_layer].shape[1]
                sys.exit()
        print "seems copacetic"
    def forward_layer(self, inputs, weights, biases, rbm_type): #completed
        if rbm_type.split('_')[1] == 'logistic':
            return self.softmax(self.weight_matrix_multiply(inputs, weights, biases))
        elif rbm_type.split('_')[1] == 'bernoulli': #rbm_type is bernoulli
            return self.sigmoid(self.weight_matrix_multiply(inputs, weights, biases))
        else:
            print "rbm_type", rbm_type, "is not a valid layer type. Valid layer types are gaussian_bernoulli, bernoulli_bernoulli, and notrbm_logistic. Exiting now..."
            sys.exit()
    def forward_pass(self, inputs, verbose=True): #completed
        # forward pass each layer starting with feature level
        if verbose:
            print "At layer 1 of", self.num_layers
        cur_layer = self.forward_layer(inputs, self.weights['weights01'], self.weights['bias1'], self.weights['weights01_type'])
        for idx in range(1,self.num_layers):
            if verbose:
                print "At layer", idx+1, "of", self.num_layers
            weight_cur_layer = ''.join(['weights',str(idx),str(idx+1)])
            weight_cur_layer_type = ''.join([weight_cur_layer,'_type'])
            bias_cur_layer = ''.join(['bias',str(idx+1)])
            cur_layer = self.forward_layer(cur_layer, self.weights[weight_cur_layer], self.weights[bias_cur_layer], self.weights[weight_cur_layer_type])
        return cur_layer
    def weight_matrix_multiply(self,inputs,weights,biases): #completed
        #print "input dims are ", inputs.shape
        #print "weight dims are ", weights.shape
        #print "bias dims are ", biases.shape
        return numpy.dot(inputs,weights)+numpy.tile(biases, (inputs.shape[0],1))
    def calculate_cross_entropy(self, output, labels): #completed
        return -numpy.sum(numpy.log([output.item((x,labels[x])) for x in range(labels.size)]))
    def calculate_classification_accuracy(self, output, labels): #completed
        prediction = output.argmax(axis=1).reshape(labels.shape)
        classification_accuracy = sum(prediction == labels) / float(labels.size)
        return classification_accuracy[0]
    #math functions
    def sigmoid(self,inputs): #completed
        return 1/(1+numpy.exp(-inputs)) #1/(1+e^-X)
    def softmax(self, inputs): #completed
        #subtracting max value of each data point below for numerical stability
        exp_inputs = numpy.exp(inputs - numpy.transpose(numpy.tile(numpy.max(inputs,axis=1), (inputs.shape[1],1))))
        return exp_inputs / numpy.transpose(numpy.tile(numpy.sum(exp_inputs, axis=1), (exp_inputs.shape[1],1)))
                
class NN_Tester(Neural_Network): #completed
    def __init__(self, config_dictionary): #completed
        #runs DNN tester soup to nuts.
        # variables are
        # feature_file_name - name of feature file to load from
        # weight_matrix_name - initial weight matrix to load
        # output_name - output predictions
        # label_file_name - label file to check accuracy
        # required are feature_file_name, weight_matrix_name, and output_name
        self.mode = 'test'
        super(NN_Tester,self).__init__(config_dictionary)
        self.check_keys(config_dictionary)
        
        self.weight_matrix_name = self.default_variable_define(config_dictionary, 'weight_matrix_name', arg_type='string')
        self.open_weight_matrix()
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
    def open_weight_matrix(self): #completed
        #this is a bit different than the NN_Trainer version, since you need correct weights to work.
        #In NN_trainer version, we initialize weights randomly if not found, don't want that here.
        extra_keys = 0
        try:
            self.weights = sp.loadmat(self.weight_matrix_name)
        except IOError:
            print "Unable to open", self.weight_matrix_name, "exiting now"
            sys.exit()
        else:
            if '__globals__' in self.weights:
                extra_keys += 1
            if '__header__' in self.weights:
                extra_keys += 1
            if '__version__' in self.weights:
                extra_keys += 1
            self.num_layers = (len(self.weights) - 1 - extra_keys) / 3 # we have biases for visible, input, output units (= num_layers + 1), weight transitions (= num_layers) and rbm_type (= num_layers)
            for idx in range(self.num_layers): #changes weight layer type to ascii string, which is what we'll need for later functions
                weight_cur_layer = ''.join(['weights',str(idx),str(idx+1)])
                weight_cur_layer_type = ''.join([weight_cur_layer, '_type'])
                self.weights[weight_cur_layer_type] = self.weights[weight_cur_layer_type][0].encode('ascii', 'ignore')
            self.check_weights()
    def write_posterior_prob_file(self): #completed
        try:
            print "Writing to", self.output_name
            sp.savemat(self.output_name,{'targets' : self.posterior_probs}, oned_as='column') #output name should have .mat extension
        except IOError:
            print "Unable to write to ", self.output_name, "... Exiting now"
            sys.exit()
        
class NN_Trainer(Neural_Network):
    def __init__(self,config_dictionary): #completed
        #variables in NN_trainer object are:
        #mode (set to 'train')
        #feature_file_name - inherited from Neural_Network class, name of feature file (in .mat format with variable 'features' in it) to read from
        #features - inherited from Neural_Network class, features
        #label_file_name - name of label file (in .mat format with variable 'labels' in it) to read from
        #labels - labels for backprop
        #architecture - specified by n_hid, n_hid, ..., n_hid. # of feature dimensions and # of classes need not be specified
        #weight_matrix_name - initial weight matrix, if specified, if not, will initialize from random
        #initial_weight_max - needed if initial weight matrix not loaded
        #initial_weight_min - needed if initial weight matrix not loaded
        #initial_bias_max - needed if initial weight matrix not loaded
        #initial_bias_min - needed if initial weight matrix not loaded
        #do_pretrain - set to 1 or 0 (probably should change to boolean values)
        #pretrain_method - not yet implemented, will either be 'mean_field' or 'sampling'
        #pretrain_iterations - # of iterations per RBM. Must be equal to the number of hidden layers
        #pretrain_learning_rate - learning rate for each epoch of pretrain. must be equal to # hidden layers * sum(pretrain_iterations)
        #pretrain_batch_size - batch size for pretraining
        #do_backprop - do backpropagation (set to either 0 or 1, probably should be changed to boolean value)
        #backprop_method - either 'steepest_descent', 'conjugate_gradient', or '2nd_order', latter two not yet implemented
        #steepest_learning_rate - learning rate for steepest_descent backprop
        #backprop_batch_size - batch size for backprop
        #output_name - name of weight file to store to.
        # At bare minimum, you'll need these variables set to train
        # feature_file_name
        # output_name
        # this will run logistic regression using steepest descent, which is a bad idea
        self.mode = 'train'
        super(NN_Trainer,self).__init__(config_dictionary)
        self.check_keys(config_dictionary)
        #read label file
        self.label_file_name = self.default_variable_define(config_dictionary, 'label_file_name', arg_type='string', error_string="No label_file_name defined, can only do pretraining",exit_if_no_default=False)
        if self.label_file_name != None:
            self.labels = self.read_label_file()
            self.check_labels()
        else:
            del self.label_file_name
        
        #read architecture
        self.architecture = self.default_variable_define(config_dictionary, 'architecture', arg_type='int_comma_string')
        
        #initialize weights
        self.weight_matrix_name = self.default_variable_define(config_dictionary, 'weight_matrix_name', exit_if_no_default=False)
        if self.weight_matrix_name == None:
            del self.weight_matrix_name
        self.initial_weight_max = self.default_variable_define(config_dictionary, 'initial_weight_max', arg_type='float', default_value=0.1)
        self.initial_weight_min = self.default_variable_define(config_dictionary, 'initial_weight_min', arg_type='float', default_value=-0.1)
        self.initial_bias_max = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=-2.2)
        self.initial_bias_min = self.default_variable_define(config_dictionary, 'initial_bias_max', arg_type='float', default_value=-2.4)
        self.open_weight_matrix()
        
        #pretraining configuration
        self.do_pretrain = self.default_variable_define(config_dictionary, 'do_pretrain', default_value=False, arg_type='boolean')
        if self.do_pretrain:
            self.pretrain_method = self.default_variable_define(config_dictionary, 'pretrain_method', default_value='mean_field', acceptable_values=['mean_field', 'sampling'])
            self.pretrain_iterations = self.default_variable_define(config_dictionary, 'pretrain_iterations', default_value=[5] * len(self.architecture), 
                                                                    error_string="No pretrain_iterations defined, setting pretrain_iterations to default 5 per layer", 
                                                                    arg_type='int_comma_string')

            weight_last_layer_type = ''.join(['weights', str(self.num_layers-1), str(self.num_layers), '_type'])
            if self.weights[weight_last_layer_type] == 'notrbm_logistic' and (len(self.pretrain_iterations) != self.num_layers - 1):
                print "given layer type", weight_last_layer_type, "pretraining iterations length should be", self.num_layers-1, "but pretraining_iterations is length ", len(self.pretrain_iterations), "... Exiting now"
                sys.exit()
            elif self.weights[weight_last_layer_type] != 'notrbm_logistic' and (len(self.pretrain_iterations) != self.num_layers):
                print "given layer type", weight_last_layer_type, "pretraining iterations length should be", self.num_layers, "but pretraining_iterations is length ", len(self.pretrain_iterations), "... Exiting now"
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
                                                                acceptable_values=['steepest_descent', 'conjugate_gradient'])
            if self.backprop_method == 'steepest_descent':
                self.steepest_learning_rate = self.default_variable_define(config_dictionary, 'steepest_learning_rate', default_value=[0.008, 0.004, 0.002, 0.001], arg_type='float_comma_string')
            elif self.backprop_method == 'conjugate_gradient':
                self.conjugate_max_iterations = self.default_variable_define(config_dictionary, 'conjugate_max_iterations', default_value=3, 
                                                                             arg_type='int')
                self.conjugate_const_type = self.default_variable_define(config_dictionary, 'conjugate_const_type', arg_type='string', default_value='polak-ribiere')
                self.conjugate_num_epochs = self.default_variable_define(config_dictionary, 'conjugate_num_epochs', default_value=20, arg_type='int')
                self.conjugate_num_line_searches = self.default_variable_define(config_dictionary, 'conjugate_num_line_searches', default_value=20, arg_type='int')
            self.backprop_batch_size = self.default_variable_define(config_dictionary, 'backprop_batch_size', default_value=2048, arg_type='int')
        self.dump_config_vals()
    def forward_first_order_methods(self, inputs): #need to check
        #returns hidden values for each layer, needed for steepest descent and conjugate gradient methods
        hiddens = {}
        hiddens[0] = inputs
        for layer_num in range(1,self.num_layers+1): #will need for steepest descent for first direction
            weight_cur_layer = ''.join(['weights',str(layer_num-1),str(layer_num)])
            weight_cur_layer_type = ''.join([weight_cur_layer,'_type'])
            bias_cur_layer = ''.join(['bias',str(layer_num)])
            hiddens[layer_num] = self.forward_layer(hiddens[layer_num-1], self.weights[weight_cur_layer], 
                                                    self.weights[bias_cur_layer], self.weights[weight_cur_layer_type] )
        return hiddens
    def update_weights(self, step_size, direction):
        #a pretty daft way of updating weights, since we have to store the entire direction (which is another copy of the weights),
        #but this is probably a smart thing to do for conjugate gradient methods for which there is no structure to exploit
        for layer_num in range(1,self.num_layers+1):
            weight_cur_layer = ''.join(['weights',str(layer_num-1),str(layer_num)])
            bias_cur_layer = ''.join(['bias',str(layer_num)])
            self.weights[weight_cur_layer] += step_size * direction[weight_cur_layer]
            self.weights[bias_cur_layer] += step_size * direction[bias_cur_layer]
    def train(self): #completed
        if self.do_pretrain:
            self.pretrain()
        if self.do_backprop:
            if self.backprop_method == 'steepest_descent':
                self.backprop_steepest_descent()
            elif self.backprop_method == 'conjugate_gradient':
                self.backprop_conjugate_gradient()
        self.write_weight_matrix()
    #pretraining functions
    def backward_layer(self, hiddens, weights, biases, rbm_type): #completed
        if rbm_type.split('_')[0] == 'gaussian':
            return self.weight_matrix_multiply(hiddens, numpy.transpose(weights), biases)
        else: #rbm_type is bernoulli
            return self.sigmoid(self.weight_matrix_multiply(hiddens, numpy.transpose(weights), biases))
    def pretrain(self): #completed
        print "starting pretraining"
        learning_rate_index = 0;
        for layer_num in range(len(self.pretrain_iterations)):
            print "pretraining rbm", layer_num+1, "of", len(self.pretrain_iterations)
            for iteration in range(self.pretrain_iterations[layer_num]):
                print "at iteration", iteration+1, "of", self.pretrain_iterations[layer_num]
                batch_index = 0
                end_index = 0
                reconstruction_error = 0
                while end_index < self.features.shape[0]: #run through batches
                    end_index = min(batch_index+self.pretrain_batch_size,self.features.shape[0])
                    batch_size = len(range(batch_index,end_index))
                    inputs = self.features[batch_index:end_index]
                    for idx in range(layer_num): #propagate to current pre-training layer
                        bias_cur_layer = ''.join(['bias', str(idx+1)])
                        weight_cur_layer = ''.join(['weights', str(idx), str(idx+1)])
                        weight_cur_layer_type = ''.join([weight_cur_layer, '_type'])
                        inputs = self.forward_layer(inputs, self.weights[weight_cur_layer], self.weights[bias_cur_layer], self.weights[weight_cur_layer_type])
                
                    bias_cur_layer  = ''.join(['bias', str(layer_num+1)])
                    bias_prev_layer = ''.join(['bias', str(layer_num)])
                    weight_cur_layer = ''.join(['weights', str(layer_num), str(layer_num+1)])
                    weight_cur_layer_type = ''.join([weight_cur_layer, '_type'])
                    
                    hiddens = self.forward_layer(inputs, self.weights[weight_cur_layer], self.weights[bias_cur_layer], self.weights[weight_cur_layer_type]) #ndata x nhid
                    reconstruction = self.backward_layer(hiddens, self.weights[weight_cur_layer], self.weights[bias_prev_layer], self.weights[weight_cur_layer_type]) #ndata x nvis
                    reconstruction_hiddens = self.forward_layer(reconstruction, self.weights[weight_cur_layer], self.weights[bias_cur_layer], self.weights[weight_cur_layer_type]) #ndata x nhid
                
                    #update weights
                    weight_update =  numpy.dot(numpy.transpose(reconstruction),reconstruction_hiddens) - numpy.dot(numpy.transpose(inputs),hiddens) # inputs: [batch_size * n_dim], hiddens: [batch_size * n_hids]
                    vis_bias_update =  numpy.sum(reconstruction, axis=0) - numpy.sum(inputs, axis=0)
                    hid_bias_update =  numpy.sum(reconstruction_hiddens, axis=0) - numpy.sum(hiddens, axis=0)
                    self.weights[weight_cur_layer] -= self.pretrain_learning_rate[learning_rate_index] / batch_size * weight_update
                    self.weights[bias_prev_layer] -= self.pretrain_learning_rate[learning_rate_index] / batch_size * vis_bias_update
                    self.weights[bias_cur_layer] -= self.pretrain_learning_rate[learning_rate_index] / batch_size * hid_bias_update
                    
                    reconstruction_error += numpy.sum((inputs - reconstruction) * (inputs - reconstruction))
                    batch_index += self.pretrain_batch_size
                print "squared reconstuction error is", reconstruction_error
                learning_rate_index += 1
    #fine-tuning/backprop functions
    def backprop_steepest_descent(self): #running, but with math problem?... I think it's fixed
        print "starting backprop using steepest descent"
        batch_index = 0
        print "Number of layers is", self.num_layers
        for epoch_num in range(len(self.steepest_learning_rate)):
            print "At epoch", epoch_num+1, "of", len(self.steepest_learning_rate), "with learning rate", self.steepest_learning_rate[epoch_num]
            cross_entropy = 0
            num_corr_classified = 0
            batch_index = 0
            end_index = 0
            num_training_examples = 0
            while end_index < self.features.shape[0]: #run through the batches
                per_done = float(batch_index)/self.features.shape[0]*100
                sys.stdout.write("\r%.1f%% done" % per_done), sys.stdout.flush()
                end_index = min(batch_index+self.backprop_batch_size,self.features.shape[0])
                batch_size = len(range(batch_index,end_index))
                hiddens = self.forward_first_order_methods(self.features[batch_index:end_index])
                cross_entropy += self.calculate_cross_entropy(hiddens[self.num_layers], self.labels[batch_index:end_index])
                num_corr_classified += int(self.calculate_classification_accuracy(hiddens[self.num_layers], self.labels[batch_index:end_index]) * batch_size)
                num_training_examples += batch_size
                #calculating negative gradient of log softmax
                weight_vec = -hiddens[self.num_layers] #batchsize x n_outputs
                for label_index in range(batch_index,end_index):
                    data_index = label_index - batch_index
                    weight_vec[data_index, self.labels[label_index]] += 1
                #averaging batches
                weight_update = numpy.dot(numpy.transpose(hiddens[self.num_layers-1]), weight_vec)
                #I don't use calculate_negative_gradient because structure allows me to store only one layer of weights
                bias_update = sum(weight_vec)
                for layer_num in range(self.num_layers-1,0,-1):
                    weight_cur_layer = ''.join(['weights',str(layer_num-1),str(layer_num)])
                    weight_next_layer = ''.join(['weights',str(layer_num),str(layer_num+1)])
                    bias_cur_layer = ''.join(['bias',str(layer_num)])
                    bias_next_layer = ''.join(['bias',str(layer_num+1)])
                    weight_vec = numpy.dot(weight_vec, numpy.transpose(self.weights[weight_next_layer])) * hiddens[layer_num] * (1-hiddens[layer_num]) #n_hid x n_out * (batchsize x n_out), do the biases get involved in this calculation???
                    
                    self.weights[weight_next_layer] += self.steepest_learning_rate[epoch_num] / batch_size * weight_update
                    self.weights[bias_next_layer] += self.steepest_learning_rate[epoch_num] / batch_size * bias_update
                    #self.weights[weight_cur_layer] += self.steepest_learning_rate[epoch_num] / batch_size * numpy.dot(numpy.transpose(hiddens[layer_num-1]), weight_vec)
                    #self.weights[bias_cur_layer] += self.steepest_learning_rate[epoch_num] / batch_size * sum(weight_vec)
                    weight_update = numpy.dot(numpy.transpose(hiddens[layer_num-1]), weight_vec)
                    bias_update = sum(weight_vec)
                #do final weight_update
                self.weights[weight_cur_layer] += self.steepest_learning_rate[epoch_num] / batch_size * weight_update
                self.weights[bias_cur_layer] += self.steepest_learning_rate[epoch_num] / batch_size * bias_update

                batch_index += self.backprop_batch_size
            sys.stdout.write("\r100.0% done\r")
            print "average cross entropy at the end of epoch is", cross_entropy / num_training_examples
            print "number correctly classified is", num_corr_classified, "of", num_training_examples
    def backprop_conjugate_gradient(self): #Running... needs better interpolation methods
        #in this framework "points" are self.weights
        #will also need to store CG-direction, which will be in dictionary conj_grad_dir
        #num_batches = int(numpy.ceil(self.features.shape[0] / float(self.backprop_batch_size)))
        print "Starting backprop using conjugate gradient"
        #print "With", self.conjugate_num_epochs, "epochs and", num_batches, "batches per epoch"
        #print num_batches * "."
        print "Number of layers is", self.num_layers
        
        for epoch_num in range(self.conjugate_num_epochs):
            print "Epoch", epoch_num+1, "of", self.conjugate_num_epochs
            cross_entropy = 0
            num_corr_classified = 0
            batch_index = 0
            end_index = 0
            num_training_examples = 0
            while end_index < self.features.shape[0]: #run through the batches
                per_done = float(batch_index)/self.features.shape[0]*100
                sys.stdout.write("\r%.1f%% done" % per_done), sys.stdout.flush()
                #sys.stdout.write("."), sys.stdout.flush()
                end_index = min(batch_index+self.backprop_batch_size,self.features.shape[0])
                batch_size = len(range(batch_index,end_index))
                batch_inputs = self.features[batch_index:end_index]
                batch_labels = self.labels[batch_index:end_index]
                
                ########## perform conjugate gradient on the batch ########################
                failed_line_search = False
                conj_grad_dir = self.calculate_negative_gradient(batch_inputs, batch_labels) #steepest descent for first direction
                old_gradient = conj_grad_dir
                #init_step_size = 1000/4 / (1 + self.dot(conj_grad_dir, conj_grad_dir))
                #print "max_step_size is", max_step_size
                for _ in range(self.conjugate_max_iterations):
                    step_size = self.line_search(batch_inputs, batch_labels, conj_grad_dir, 
                                                 max_line_searches=self.conjugate_num_line_searches, 
                                                 zero_step_directional_derivative=-self.dot(old_gradient, conj_grad_dir))
                    if step_size > 0: #line search did not fail
                        #no need to update weights here, because we've been updating it during the line search
                        failed_line_search = False
                        #update search direction
                        new_gradient = self.calculate_gradient(batch_inputs, batch_labels)
                        conj_grad_dir = self.calculate_conjugate_gradient_direction(batch_inputs, batch_labels, old_gradient, new_gradient, conj_grad_dir, 
                                                                                    const_type=self.conjugate_const_type)
                        del old_gradient
                        old_gradient = new_gradient
                        del new_gradient
                        if self.dot(conj_grad_dir, conj_grad_dir) > 0: #conjugate gradient direction not a descent direction, switching to steepest descent
                            conj_grad_dir = self.calculate_negative_gradient(batch_inputs, batch_labels)
                            old_gradient = conj_grad_dir
                    else: #line search failed
                        if failed_line_search: #failed line search twice in a row, so bail
                            break
                        failed_line_search = True
                        conj_grad_dir = self.calculate_negative_gradient(batch_inputs, batch_labels)
                        old_gradient = conj_grad_dir
                ###########end conjugate gradient batch ####################################
                
                batch_targets = self.forward_pass(batch_inputs, verbose=False)
                cross_entropy += self.calculate_cross_entropy(batch_targets, batch_labels)
                #print "cross entropy in batch is", self.calculate_cross_entropy(batch_targets, batch_labels)
                num_corr_classified += int(self.calculate_classification_accuracy(batch_targets, batch_labels) * batch_size)
                num_training_examples += batch_size
                batch_index += self.backprop_batch_size
            sys.stdout.write("\r100.0% done\r")
            print "average cross entropy at the end of epoch is", cross_entropy / num_training_examples
            print "number correctly classified is", num_corr_classified, "of", num_training_examples
    def calculate_conjugate_gradient_direction(self, batch_inputs, batch_labels, old_gradient, new_gradient, 
                                               current_conjugate_gradient_direction, const_type='polak-ribiere'):
        new_conjugate_gradient_direction = {}
        if const_type == 'polak-ribiere':
            cg_const = (self.dot(new_gradient, new_gradient) - self.dot(new_gradient,old_gradient)) / self.dot(old_gradient, old_gradient)
        elif const_type == 'fletcher-reeves':
            cg_const = self.dot(new_gradient, new_gradient) / self.dot(old_gradient, old_gradient)
        elif const_type == 'polak-ribiere+':
            cg_const = max((self.dot(new_gradient, new_gradient) - self.dot(new_gradient,old_gradient)) / self.dot(old_gradient, old_gradient),0)
        elif const_type == 'hestenes-stiefel': #might run into numerical stability issues
            cg_const = (self.dot(new_gradient, new_gradient) - self.dot(new_gradient, old_gradient)) / max((self.dot(new_gradient, current_conjugate_gradient_direction) - self.dot(old_gradient, current_conjugate_gradient_direction)), 1E-4)
        else:
            print const_type, "not recognized, the only valid methods of \'polak-ribiere\', \'fletcher-reeves\', \'polak-ribiere+\', \'hestenes-stiefel\'... Exiting now"
            sys.exit()
        
        for layer_num in range(1,self.num_layers+1):
            weight_cur_layer = ''.join(['weights',str(layer_num-1),str(layer_num)])
            bias_cur_layer = ''.join(['bias',str(layer_num)])
            new_conjugate_gradient_direction[weight_cur_layer] = -new_gradient[weight_cur_layer] + cg_const * current_conjugate_gradient_direction[weight_cur_layer]
            new_conjugate_gradient_direction[bias_cur_layer] = -new_gradient[bias_cur_layer] + cg_const * current_conjugate_gradient_direction[bias_cur_layer]
        
        return new_conjugate_gradient_direction
    def calculate_gradient(self, batch_inputs, batch_labels): #want to make this more general to handle arbitrary loss functions, structures
        gradient = {}
        batch_size = batch_labels.size
        
        hiddens = self.forward_first_order_methods(batch_inputs)
        
        #derivative of log(cross-entropy softmax)
        weight_vec = hiddens[self.num_layers] #batchsize x n_outputs
        for index in range(batch_size):
            weight_vec[index, batch_labels[index]] -= 1
        
        #average layers in batch
        weight_cur_layer = ''.join(['weights',str(self.num_layers-1),str(self.num_layers)])
        bias_cur_layer = ''.join(['bias',str(self.num_layers)])
        gradient[weight_cur_layer] = numpy.dot(numpy.transpose(hiddens[self.num_layers-1]), weight_vec) 
        gradient[bias_cur_layer] = sum(weight_vec)
        
        #propagate to sigmoid layers
        for layer_num in range(self.num_layers-1,0,-1):
            weight_cur_layer = ''.join(['weights',str(layer_num-1),str(layer_num)])
            weight_next_layer = ''.join(['weights',str(layer_num),str(layer_num+1)])
            bias_cur_layer = ''.join(['bias',str(layer_num)])
            weight_vec = numpy.dot(weight_vec, numpy.transpose(self.weights[weight_next_layer])) * hiddens[layer_num] * (1-hiddens[layer_num]) #n_hid x n_out * (batchsize x n_out)

            gradient[weight_cur_layer] = numpy.dot(numpy.transpose(hiddens[layer_num-1]), weight_vec)
            gradient[bias_cur_layer] = sum(weight_vec)
        
        return gradient
    def calculate_negative_gradient(self, batch_inputs, batch_labels):
        negative_gradient = self.calculate_gradient(batch_inputs, batch_labels)
        for layer_num in range(1, self.num_layers+1):
            weight_cur_layer = ''.join(['weights',str(layer_num-1),str(layer_num)])
            bias_cur_layer = ''.join(['bias',str(layer_num)])
            negative_gradient[weight_cur_layer] *= -1
            negative_gradient[bias_cur_layer] *= -1
        
        return negative_gradient
    def dot(self, weight1, weight2): #completed in theory
        #An abuse of terminology, I consider two weight dictionaries to be weight vectors, 
        #so dot produces the "dot product" of the two weight dictionaries
        return_val = 0
        for layer_num in range(1,self.num_layers+1):
            weight_cur_layer = ''.join(['weights',str(layer_num-1),str(layer_num)])
            bias_cur_layer = ''.join(['bias',str(layer_num)])
            return_val += numpy.sum(weight1[weight_cur_layer] * weight2[weight_cur_layer])
            return_val += numpy.sum(weight1[bias_cur_layer] * weight2[bias_cur_layer])
        return return_val
    def line_search(self, batch_inputs, batch_labels, direction, max_step_size=0.1,
                    max_line_searches=20, zero_step_directional_derivative=None):
        # the line search algorithm is basically as follows
        # we have directional derivative of p_k at cross_entropy(0), in gradient_direction, c_1, and c_2, and stepsize_max, current cross-entropy in batch
        # choose stepsize to be between 0 and stepsize_max (usually by finding minimum or quadratic, cubic, or quartic function)
        #while loop
        #    evaluate cross-entropy at point weight + stepsize * gradient direction
        #    if numerical issue (cross_entropy is inf, etc).
        #        divide step_size by 2 and try again
        #    if fails first Wolfe condition (i.e., evaluated_cross_entropy > current_cross_entropy + c_1 * stepsize * dir_deriv(cross_ent(0))
        #        interpolate between (prev_stepsize, cur_stepsize) and return that stepsize #we went too far in the current direction
        #    #if not we made it past first Wolfe condition
        #    calculate directional derivative at proposed point
        #    if made it past second Wolfe condition (i.e., abs(prop_dir_deriv) <= -c_2 dir_deriv(0)
        #        finished line search
        #    elif dir_deriv(proposed) >= 0 #missed minimum before
        #        interp between current step size and previous one
        #     otherwise we essentially didn't go far enough with our step size, so find step_size between current_stepsize and max_stepsize
        c_1 = 1E-4
        c_2 = 0.9
        zero_step_loss = self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False), batch_labels) #\phi_0
        if zero_step_directional_derivative is None: #no need to calculation directional derivative if already passed
            zero_step_directional_derivative = self.dot(self.calculate_gradient(batch_inputs, batch_labels), direction)

        proposed_step_size = max_step_size / 2
        prev_step_size = 0
        prev_loss = zero_step_loss
        prev_directional_derivative = zero_step_directional_derivative
        
        overall_step_size = 0
        (upper_bracket, upper_bracket_loss, upper_bracket_deriv, lower_bracket, lower_bracket_loss, lower_bracket_deriv) = [0 for _ in range(6)]
        
        for num_line_searches in range(1,max_line_searches+1): #looking for brackets
            #update weights
            #print proposed_step_size, ",", prev_step_size, ",", zero_step_loss, ",",
            self.update_weights(proposed_step_size - prev_step_size, direction)
            overall_step_size += proposed_step_size - prev_step_size #keep track of where we moved along the line search
            proposed_loss = self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False), batch_labels)
            #print proposed_loss
            if math.isinf(proposed_loss) or math.isnan(proposed_loss): #numerical stability issues
                print "have numerical stability issues, so decreasing step size by 1/2"
                prev_step_size = proposed_step_size #restore weights
                proposed_step_size /= 2
                continue
            if proposed_loss > zero_step_loss + c_1 * proposed_step_size * zero_step_directional_derivative: #fails Armijo rule, but we have found our bracket
                # we now know that Wolfe conditions are satisfied between prev_step_size  and proposed_step_size
                #print "Armijo rule failed, so generating brackets"
                upper_bracket = proposed_step_size
                upper_bracket_loss = proposed_loss
                upper_bracket_deriv = self.dot(self.calculate_gradient(batch_inputs, batch_labels), direction)
                lower_bracket = prev_step_size
                lower_bracket_loss = prev_loss
                lower_bracket_deriv = prev_directional_derivative
                prev_step_size = proposed_step_size
                break
            proposed_directional_derivative = self.dot(self.calculate_gradient(batch_inputs, batch_labels), direction)
            if abs(proposed_directional_derivative) <= -c_2 * zero_step_directional_derivative: #satisfies strong Wolfe condition
                #print "Wolfe conditions satisfied"
                return proposed_step_size
            elif proposed_directional_derivative >= 0:
                #print "went too far for second order condition, brackets found"
                lower_bracket = proposed_step_size
                lower_bracket_loss = proposed_loss
                lower_bracket_deriv = proposed_directional_derivative
                upper_bracket = prev_step_size
                upper_bracket_loss = prev_loss
                lower_bracket_deriv = prev_directional_derivative
                prev_step_size = proposed_step_size
                break
            else: #satisfies Armijo rule, but not 2nd Wolfe condition, so go out further
                #print "satisfied Armijo, but not Wolfe, so increasing step size. Current loss is", zero_step_loss, "and proposed loss is", proposed_loss
                #print "derivative of step size is", proposed_directional_derivative, "current derivative is", cur_directional_derivative
                #extrapolated_step_size = proposed_step_size * max_step_size_jump #need to change to another interpolation method
                #print "extrapolated step size", extrapolated_step_size
                #a = 6 * (prev_loss - proposed_loss) + 3 * (prev_directional_derivative + proposed_directional_derivative) * (proposed_step_size - prev_step_size)
                #b = 3 * (proposed_loss - prev_loss) - (2 * prev_directional_derivative + proposed_directional_derivative) * (proposed_step_size - prev_step_size)
                #extrapolated_step_size = prev_step_size - prev_directional_derivative * (proposed_step_size - prev_step_size) ** 2 / (b + numpy.sqrt(b**2 - a * prev_directional_derivative * (proposed_step_size - prev_step_size)))
                #if math.isnan(extrapolated_step_size) or math.isinf(extrapolated_step_size): #numerical problem
                #    print "numerical issue with extrapolation, so extrapolating max amount"
                #    extrapolated_step_size = max_step_size_jump * prev_step_size
                #elif extrapolated_step_size > max_step_size_jump * prev_step_size: #went out too far
                #    extrapolated_step_size = max_step_size_jump * prev_step_size
                #elif (extrapolated_step_size - proposed_step_size) / (proposed_step_size - prev_step_size) < min_step_size_jump:
                    #print "step size too close, moving step size out"
                #    extrapolated_step_size = proposed_step_size + (proposed_step_size - prev_step_size) * min_step_size_jump
                prev_step_size = proposed_step_size
                prev_loss = proposed_loss
                prev_directional_derivative = proposed_directional_derivative
                proposed_step_size = (prev_step_size + max_step_size) / 2
                
        #after first loop, weights are set prev_step_size * direction, with upper and lower brackets set, now find step size that
        #satisfy Wolfe conditions
        remaining_line_searches = max_line_searches - num_line_searches
        
        for _ in range(remaining_line_searches): #searching for good step sizes within bracket
            proposed_step_size = self.interpolate_step_size((upper_bracket, upper_bracket_loss, upper_bracket_deriv),
                                                        (lower_bracket, lower_bracket_loss, lower_bracket_deriv))#(upper_bracket + lower_bracket) / 2 #need to change to another interpolation method
            #print "proposed step size is", proposed_step_size, "while upper bracket is", upper_bracket, "and lower bracket is", lower_bracket
            self.update_weights(proposed_step_size - prev_step_size, direction)
            overall_step_size += proposed_step_size - prev_step_size
            proposed_loss = self.calculate_cross_entropy(self.forward_pass(batch_inputs, verbose=False), batch_labels)
            proposed_directional_derivative = self.dot(self.calculate_gradient(batch_inputs, batch_labels), direction)
            #print "proposed loss is", proposed_loss, "and dir_deriv", proposed_directional_derivative
            #print "zero_step_dir_deriv", zero_step_directional_derivative
            if proposed_loss > zero_step_loss + c_1 * proposed_step_size * zero_step_directional_derivative:
                #print "Armijo rule failed, adjusting brackets"
                upper_bracket = proposed_step_size
                upper_bracket_loss = proposed_loss
                upper_bracket_deriv = proposed_directional_derivative
                prev_step_size = proposed_step_size
            else:
                #print "Armijo rule satisfied"
                if abs(proposed_directional_derivative) <= -c_2 * zero_step_directional_derivative: #satisfies strong Wolfe condition
                    #print "Satisfied Wolfe conditions, step size is", proposed_step_size
                    return proposed_step_size
                elif proposed_directional_derivative * (upper_bracket - lower_bracket) >= 0:
                    #print "went too far on step ... adjusting brackets"
                    upper_bracket = lower_bracket
                    upper_bracket_loss = lower_bracket_loss
                    upper_bracket_deriv = lower_bracket_deriv
                    lower_bracket = proposed_step_size
                    lower_bracket_loss = proposed_loss
                    lower_bracket_deriv = proposed_directional_derivative
                    prev_step_size = proposed_step_size
        
        #if we made it this far, we ran out of line searches and line_search failed
        print "\nline search failed, so restoring weights..."
        self.update_weights(-overall_step_size, direction) #restore previous weights
        return 0 #line search failed
    def interpolate_step_size(self, p1, p2):
        #p1 and p2 are tuples in form (x,f(x), f'(x)) and spits out minimum step size based on cubic interpolation of data
        if abs(p2[0] - p1[0]) < 1E-4:
            print "difference between two step sizes is small |p2 -p1|", abs(p2[0] - p1[0]), "returning bisect"
            return (p1[0] + p2[0]) / 2
        b = (3 * (p2[1] - p1[1]) - (2 * p1[2] + p2[2])*(p2[0] - p1[0])) / (p2[0] - p1[0])**2
        a = (-2 * (p2[1] - p1[1]) + (p1[2] + p2[2])*(p2[0] - p1[0])) / (p2[0] - p1[0])**3
        if b**2 > 3 * a * p1[2] and math.isinf(a) == False and math.isnan(a) == False: #cubic interp
            return p1[0] + (-b + numpy.sqrt(b**2 - 3 * a * p1[2])) / (3 * a)
        elif b != 0: #quadratic interp
            return p1[0] - p1[2] / (2 * b)
        else: #bisect
            return (p1[0] + p2[0]) / 2
    def open_weight_matrix(self): #completed
        extra_keys = 0
        try:    
            self.weights = sp.loadmat(self.weight_matrix_name)
        except IOError:
            print "weight_matrix_name", self.weight_matrix_name, "not found, so initializing weights...",
            self.init_weight_matrix()
        except AttributeError:
            print "\'weight_matrix_name\' (for initial weights) not defined, so initializing weights...",
            self.init_weight_matrix()
        else:
            print self.weight_matrix_name, " found, initializing weights and ignoring architecture value" #want to redefine architecture value
            if '__globals__' in self.weights:
                extra_keys += 1
            if '__header__' in self.weights:
                extra_keys += 1
            if '__version__' in self.weights:
                extra_keys += 1
            self.num_layers = (len(self.weights) - 1 - extra_keys) / 3 # we have biases for visible, input, output units (= num_layers + 1), weight transitions (= num_layers) and rbm_type (= num_layers)
            for idx in range(self.num_layers): #changes weight layer type to ascii string, which is what we'll need for later functions
                weight_cur_layer = ''.join(['weights',str(idx),str(idx+1)])
                weight_cur_layer_type = ''.join([weight_cur_layer, '_type'])
                self.weights[weight_cur_layer_type] = self.weights[weight_cur_layer_type][0].encode('ascii', 'ignore')
            self.check_weights()
    def init_weight_matrix(self): #completed
        self.weights = {}
        self.num_layers = len(self.architecture)
        initial_bias_range = self.initial_bias_max - self.initial_bias_min
        initial_weight_range = self.initial_weight_max - self.initial_weight_min
        self.weights['bias0'] = (self.initial_bias_min + 
                                initial_bias_range * 
                                numpy.random.random_sample((1,self.features.shape[1])))
        
        for idx in range(len(self.architecture)):
            weight_cur_layer = ''.join(['weights',str(idx),str(idx+1)])
            weight_cur_layer_type = ''.join([weight_cur_layer,'_type'])
            bias_cur_layer = ''.join(['bias',str(idx+1)])
            self.weights[bias_cur_layer] = (self.initial_bias_min + 
                                            initial_bias_range * 
                                            numpy.random.random_sample((1,self.architecture[idx])))
            if idx == 0:
                self.weights[weight_cur_layer_type] = 'gaussian_bernoulli'
                self.weights[weight_cur_layer]=(self.initial_weight_min + 
                                                initial_weight_range * 
                                                numpy.random.random_sample( (self.features.shape[1],self.architecture[idx])) )
            else:
                self.weights[weight_cur_layer_type] = 'bernoulli_bernoulli'
                self.weights[weight_cur_layer]=(self.initial_weight_min + 
                                                initial_weight_range * 
                                                numpy.random.random_sample( (self.architecture[idx-1],self.architecture[idx]) ))
        #initialize last layer
        last_layer_idx = len(self.architecture)
        weight_cur_layer = ''.join(['weights',str(last_layer_idx),str(last_layer_idx+1)])
        weight_cur_layer_type = ''.join([weight_cur_layer,'_type'])
        bias_cur_layer = ''.join(['bias',str(last_layer_idx+1)])
        
        if hasattr(self, 'labels'):
            self.num_layers += 1
            self.weights[weight_cur_layer_type] = 'notrbm_logistic'
            self.weights[bias_cur_layer] = (self.initial_bias_min + 
                                            initial_bias_range * 
                                            numpy.random.random_sample((1,numpy.max(self.labels)+1)))
            self.weights[weight_cur_layer]=(self.initial_weight_min + 
                                                initial_weight_range * 
                                                numpy.random.random_sample( (self.architecture[last_layer_idx-1],numpy.max(self.labels)+1) ))
        else:
            print "no labels found... not initializing last layer"
        print "Finished Initializing Weights"
        self.check_weights()
    def write_weight_matrix(self): #completed
        try:
            sp.savemat(self.output_name, self.weights, oned_as='column')
        except IOError:
            print "Unable to save ", self.output_name, "... Exiting now"
            sys.exit()
        else:
            print self.output_name, " successfully saved"

if __name__ == '__main__':
    script_name, config_filename = sys.argv
    print "Opening config file: %s" % config_filename
    
    try:
        config_file=open(config_filename)
    except IOError:
        print "Could open file", config_filename, ". Usage is ", script_name, "<config file>... Exiting Now"
        sys.exit()
        
    config_dictionary={}
    
    #read lines into a configuration dictionary
    for line in config_file.readlines():
        config_line=line.strip(' \n\t').split('=')
        if len(config_line[0]) > 0 and config_line[0][0] != '#': #whitespace and # symbols ignored
            config_dictionary[config_line[0]] = config_line[1]

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
                
                