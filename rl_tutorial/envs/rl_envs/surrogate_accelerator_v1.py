import gym
import os, sys
from gym import spaces
from gym.utils import seeding
import pandas as pd
import requests

from tensorflow import keras
import numpy as np

import rl_tutorial.dataprep.dataset as dp

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-FNAL-Logger')
logger.setLevel(logging.ERROR)

np.seterr(divide='ignore', invalid='ignore')

def create_dataset(dataset, look_back=10 * 15, look_forward=1):
    X, Y = [], []
    offset = look_back + look_forward
    for i in range(len(dataset) - (offset + 1)):
        xx = dataset[i:(i + look_back), 0]
        yy = dataset[(i + look_back):(i + offset), 0]
        X.append(xx)
        Y.append(yy)
    return np.array(X), np.array(Y)

def get_dataset(df, variable='B:VIMIN'):
    dataset = df[variable].values
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))

    train_size = int(len(dataset) * 0.70)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    X_train, Y_train = create_dataset(train, look_back= 15)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]))

    return X_train, Y_train

def all_inplace_scale(df):
    scale_dict = {}

    for var in ['B:VIMIN', 'B:IMINER', 'B_VIMIN', 'B:LINFRQ', 'I:IB', 'I:MDAT40']:
      our_data2 = df
      trace = our_data2[var].astype('float32')
      data = np.array(trace)
      median = np.median(data)
      upper_quartile = np.percentile(data, 75)
      lower_quartile = np.percentile(data, 25)
      iqr = upper_quartile - lower_quartile
      lower_whisker = data[data>=lower_quartile-1.5*iqr].min()
      upper_whisker = data[data<=upper_quartile+1.5*iqr].max()
      ranged = upper_whisker - lower_whisker
      our_data2[var] = 1.0/ranged*(data - median)
      
      scale_dict[str(var)] = {"median": median, "range": ranged}
    
    return scale_dict

def unscale(var_name, tseries, scale_dict):
    #equivalent to inverse transform
    from_model = np.asarray(tseries)
    update = from_model*scale_dict[str(var_name)]["range"] + scale_dict[str(var_name)]["median"]

    return(update)

def rescale(var_name, tseries, scale_dict):
    #equivalent to transform
    data = np.asarray(tseries)
    update = 1/scale_dict[str(var_name)]["range"]*(data - scale_dict[str(var_name)]["median"])

    return(update)

def create_dropout_predict_model(model, dropout):
    
    # Load the config of the original model
    conf = model.get_config()

    # Add the specified dropout to all layers
    for layer in conf['layers']:

        # Dropout layers
        if layer["class_name"] == "Dropout":
            layer["config"]["rate"] = dropout

    model_dropout = keras.Model.from_config(conf)

    model_dropout.set_weights(model.get_weights()) 
    return model_dropout

def regulation(alpha, gamma, error, min_set, beta):
  ## calculate the prediction with current regulation rules
  ER = error
  _MIN = min_set
  for i in range(len(_MIN)):
      if i>0:
            beta_t = beta[-1] + gamma*ER[i]
            beta.append(beta_t)
  MIN_pred = _MIN - alpha * ER - np.asarray(beta[-15:]).reshape(15,1)
  return MIN_pred

class Surrogate_Accelerator_v1(gym.Env):
    def __init__(self):

        self.save_dir = os.getcwd()
        self.episodes = 0
        self.steps = 0
        self.max_steps = 100
        self.total_reward = 0
        self.data_total_reward = 0
        self.diff = 0

        self.rachael_reward = 0
        self.rachael_beta = [0]

        model = keras.models.load_model('../surrogate_models/fnal_tutorial_model.h5')
        self.booster_model = create_dropout_predict_model(model, 0.0)


        # Load data to initialize the env
        # Check if data is available else download it ##################
        booster_dir = os.path.dirname(__file__)
        booster_data_file = 'BOOSTR.csv'
        booster_file_pfn = os.path.join(booster_dir, booster_data_file)
        logger.info('Booster data file pfn:{}'.format(booster_file_pfn))
        if not os.path.exists(booster_file_pfn):
            logger.info('No cached file. Downloading...')
            try:
                url = 'https://zenodo.org/record/4088982/files/data%20release.csv?download=1'
                r = requests.get(url, allow_redirects=True)
                open(booster_file_pfn, 'wb').write(r.content)
            except:
                logger.error("Problem downloading file")
        else:
            logger.info('Using exiting cached file')

        data = dp.load_reformated_cvs(booster_file_pfn, nrows=250000)
        scale_dict = all_inplace_scale(data)
        
        #################################################################

        # Preprocess the data ###########################################
        
        data['B:VIMIN'] = data['B:VIMIN'].shift(-1)
        data = data.set_index(pd.to_datetime(data.time))
        data = data.dropna()
        data = data.drop_duplicates()
        self.variables = ['B:VIMIN', 'B:IMINER', 'B_VIMIN', 'B:LINFRQ', 'I:IB', 'I:MDAT40']
        self.nvariables = len(self.variables)
        logger.info('Number of variables:{}'.format(self.nvariables))

        self.scale_dict = scale_dict
        data_list = []
        x_train = []

        # get_dataset also normalizes the data
        for v in range(len(self.variables)):
            data_list.append(get_dataset(data, variable=self.variables[v]))
            x_train.append(data_list[v][0])


        # Axis
        self.concate_axis = 1
        self.X_train = np.concatenate(x_train, axis=self.concate_axis)
        self.X_train = self.X_train[0:250,:,:]
        print("self.X_train ", self.X_train.shape)

        #sys.exit(1)
        self.nbatches = self.X_train.shape[0]
        self.nsamples = self.X_train.shape[2]
        self.batch_id = 100
        self.data_state = None
        
        ##################################################################

        print('Data shape:{}'.format(self.X_train.shape))
        self.observation_space = spaces.Box(
            low=0,
            high=+1,
            shape=(self.nvariables,),
            dtype=np.float64
        )

        # DYNAMIC ACTION SPACE SIZING ######################################
        data['B:VIMIN_DIFF'] = data['B:VIMIN'] - data['B:VIMIN'].shift(-1)
        self.nactions = 15
        self.action_space = spaces.Discrete(self.nactions)
        self.actionMap_VIMIN = []
        for i in range(1, self.nactions + 1):
            self.actionMap_VIMIN.append(data['B:VIMIN_DIFF'].quantile(i / (self.nactions + 1)))
            
        ########################################################################

        self.VIMIN = 0
        self.state = np.zeros(shape=(1, self.nvariables, self.nsamples))
        self.predicted_state = np.zeros(shape=(1, self.nvariables, 1))

        self.rachael_state = np.zeros(shape=(1, self.nvariables, self.nsamples))
        self.rachael_predicted_state = np.zeros(shape=(1, self.nvariables, 1))

        logger.debug('Init pred shape:{}'.format(self.predicted_state.shape))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.steps += 1
        logger.info('Episode/State: {}/{}'.format(self.episodes, self.steps))
        done = False

        # Steps:
        # 1) Update VIMIN based on action
        # 2) Predict booster variables
        # 3) Shift state with new values

        # Step 1: Calculate the new B:VIMIN based on policy action ######################################################
        
        logger.info('Step() before action VIMIN:{}'.format(self.VIMIN))
        delta_VIMIN = self.actionMap_VIMIN[int(action)]
        DENORN_BVIMIN = unscale(self.variables[0], np.array([self.VIMIN]).reshape(1, -1), self.scale_dict)

        DENORN_BVIMIN += delta_VIMIN
        logger.debug('Step() descaled VIMIN:{}'.format(DENORN_BVIMIN))
        logger.debug('Action:{}'.format(delta_VIMIN))

        #Rachael's Eq as an action
        alpha = 10e-2
        gamma = 7.535e-5

        B_VIMIN_trace = unscale(self.variables[2], self.state[0, 2, :].reshape(-1, 1), self.scale_dict)
        BIMINER_trace = unscale(self.variables[1], self.state[0, 1, :].reshape(-1, 1), self.scale_dict)

        self.rachael_state[0][0][self.nsamples - 1] = rescale(self.variables[0], regulation(alpha, gamma, error = BIMINER_trace, min_set = B_VIMIN_trace, beta = self.rachael_beta)[-1].reshape(-1, 1), self.scale_dict)

        self.VIMIN = rescale(self.variables[0], DENORN_BVIMIN, self.scale_dict)

        logger.debug('Step() updated VIMIN:{}'.format(self.VIMIN))
        self.state[0][0][self.nsamples - 1] = self.VIMIN
        

        # Step 2: Predict using booster model ##################################################################################
        
        self.predicted_state = self.booster_model.predict_on_batch(self.state)
        self.predicted_state = self.predicted_state.reshape(1, 2, 1)

        #Rachael's equation
        self.rachael_predicted_state = self.booster_model.predict_on_batch(self.rachael_state)
        self.rachael_predicted_state = self.rachael_predicted_state.reshape(1, 2, 1)
        
        # Step 3: Shift state by one step #######################################################################################
        
        self.state[0, :, 0:-1] = self.state[0, :, 1:]
        self.rachael_state[0, :, 0:-1] = self.rachael_state[0, :, 1:]
        # Update IMINER
        self.state[0][1][self.nsamples - 1] = self.predicted_state[0, 1:2]
        self.rachael_state[0][1][self.nsamples - 1] = self.rachael_predicted_state[0, 1:2]
        # Update data state for rendering
        self.data_state = np.copy(self.X_train[self.batch_id + self.steps].reshape(1, self.nvariables, self.nsamples))
        data_iminer = unscale(self.variables[1], self.data_state[0][1][self.nsamples - 1].reshape(1, -1), self.scale_dict)
        data_reward = -abs(data_iminer)

        # Use data for everything but the B:IMINER prediction
        self.state[0, 2:self.nvariables, :] = self.data_state[0, 2:self.nvariables, :]
        self.rachael_state[0, 2:self.nvariables, :] = self.data_state[0, 2:self.nvariables, :]

        iminer = self.predicted_state[0, 1]
        logger.debug('norm iminer:{}'.format(iminer))
        iminer = unscale(self.variables[1], np.array([iminer]), self.scale_dict).reshape(1, -1)
        logger.debug('iminer:{}'.format(iminer))

        # Reward
        reward = -abs(iminer)

        rach_reward = -abs(unscale(self.variables[1], np.array([self.rachael_predicted_state[0, 1]]).reshape(1, -1), self.scale_dict))

        if self.steps >= int(self.max_steps):
            done = True

        self.diff += np.asscalar(abs(data_iminer - iminer))
        self.data_total_reward += np.asscalar(data_reward)
        self.total_reward += np.asscalar(reward)
        self.rachael_reward += np.asscalar(rach_reward)
        
        info = {"rach_reward":np.asscalar(rach_reward), "data_reward":np.asscalar(data_reward),}

        return self.state[0, :, -1:].flatten(), np.asscalar(reward), done, info

    def reset(self):
        self.episodes += 1
        self.steps = 0
        self.data_total_reward = 0
        self.total_reward = 0
        self.diff = 0
        self.rachael_reward = 0
        self.rachael_beta = [0]

        # Prepare the random sample ##
        #self.batch_id = self.episodes + 4200
        logger.info('Resetting env')
        logger.debug('self.state:{}'.format(self.state))
        self.state = None
        self.state = np.copy(self.X_train[self.batch_id].reshape(1, self.nvariables, self.nsamples))

        self.data_state = None
        self.state = np.copy(self.X_train[self.batch_id].reshape(1, self.nvariables, self.nsamples))
        
        self.rachael_state = None
        self.rachael_state = np.copy(self.X_train[self.batch_id].reshape(1, self.nvariables, self.nsamples))

        logger.debug('self.state:{}'.format(self.state))
        logger.debug('reset_data.shape:{}'.format(self.state.shape))
        self.VIMIN = self.state[0, 0, -1:]
        logger.debug('Normed VIMIN:{}'.format(self.VIMIN))
        logger.debug('B:VIMIN:{}'.format(unscale(self.variables[0], np.array([self.VIMIN]), self.scale_dict).reshape(1, -1)))

        return self.state[0, :, -1:].flatten()

