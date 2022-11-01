# Reinforcement Learning environments and agents/policies used for the FNAL accelerator application

## Software Requirement
* Python 3.7 
* The environemnt framework is built of [OpenAI Gym](https://gym.openai.com) 
* Additional python packages are defined in the setup.py 
* For now, we assumes you are running at the top directory 

## Installing 
* Pull code from repo
```
git clone https://github.com/JeffersonLab/jlab_datascience_tutorials.git
```
* Install jlab_datascience_tutorials (via pip):
```
cd jlab_datascience_tutorials/rl_tutorial
pip install -e . --user
```

## Directory Organization
```
├── setup.py
├── drivers                           : a folder contains RL steering scripts  
├── surrogate_models                  : a folder contains surrogate model code
├── agents                            : a folder contains agent codes
├── envs                              : a folder containing the accelerator environments
├── cfg                               : a folder contains the agent and environment configuration
├── utils                             : a folder contains utilities
├── dataprep                          : a folder contains tools to prepare the FNAL booster data
          
```

## Running the example
```
cd jlab_datascience_tutorials/rl_tutorial/drivers
python run_dqn.py
```
