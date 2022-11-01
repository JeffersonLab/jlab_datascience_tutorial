# -*- coding: utf-8 -*-
import argparse
import gym
import time
import logging
import os
import random
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# Seed value
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Framework classes
import rl_tutorial.agents as agents
import rl_tutorial.envs

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Workflow-Logger')
logger.setLevel(logging.INFO)


def run_opt(nepisodes, agent_id, env_id, doPlay):

    # Create the log files to write stuff ######################
    now = datetime.now()
    timestamp = now.strftime("D%m%d%Y-T%H%M%S")
    print("date and time:", timestamp)

    lower_env_id = env_id.lower()
    lower_agent_id = agent_id.lower()
    logdir = './results/env_' + lower_env_id \
             + '_model_' + lower_agent_id \
             + '_date_' + datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        os.mkdir(logdir)
    except OSError as error:
        print(error)
    file_writer = tf.summary.create_file_writer(logdir + '/metrics')
    file_writer.set_as_default()
    
    ##############################################################


    # Train
    EPISODES = nepisodes
    best_reward = -100000
    if doPlay:
        EPISODES = 1

    # Setup environment
    estart = time.time()
    env = gym.make(env_id)
    agent_config = '../cfg/dqn_setup.json'
    NSTEPS = 50
    if "CartPole" in env_id:
        print("Setting NSTEPS to 100 for CartPole env!")
        NSTEPS: int = 100
    env._max_episode_steps = NSTEPS
    print("Env max steps: ", env._max_episode_steps)
    env.seed(seed_value)
    end = time.time()
    
    
    logger.info('Time init environment: %s' % str((end - estart) / 60.0))
    logger.info('Using environment: %s' % env)
    logger.info('Observation_space: %s' % env.observation_space.shape)
    logger.info('Action_size: %s' % env.action_space)

    # Setup the agent
    agent_id = "KerasDQN-v1"
    agent = agents.make(agent_id, env=env, cfg=agent_config)
    if doPlay:
        agent.load_model()


    #counter = 0
    ep = 0
    total_reward_list = []
    PID_reward_list = []
    data_reward_list = []
    avg_total_reward = []
    avg_PID_reward = []
    for e in tqdm(range(EPISODES), desc='RL Episodes', leave=True):
        # logger.info('Starting new episode: %s' % str(e))
        current_state = env.reset()
        total_reward = 0
        total_pid_reward = 0
        total_data_reward = 0
        step_counter = 0
        done = False
        ep += 1
        while not done:
            action, policy_type = agent.action(current_state)
            
            if doPlay:
                action, policy_type = agent.play(current_state)

            next_state, reward, done, info = env.step(action)
            store_reward = reward
            # Check if maximum episode steps is reached
            if step_counter >= NSTEPS-1:
                done = True

            if "Surrogate_Accelerator" in env_id:
                pid_reward = info['rach_reward']
                data_reward = info['data_reward']
            
            if "CartPole" in env_id:
                if done and step_counter < env._max_episode_steps-1:
                    print(done, step_counter, env._max_episode_steps-1)
                    store_reward = -100


            if not doPlay:
                agent.remember(current_state, action, store_reward, next_state, done)
                agent.train()

            # Update  current state
            current_state = next_state

            # Increment total reward
            total_reward += reward
            if "Surrogate_Accelerator" in env_id:
                total_pid_reward += float(pid_reward)
                total_data_reward += float(data_reward)

            step_counter += 1
            #counter += 1

        # Save to TensorBoard
        tf.summary.scalar('Reward', data=total_reward, step=int(ep))
        if "Surrogate_Accelerator" in env_id:
            tf.summary.scalar('PID reward', data=total_pid_reward, step=int(ep))
            tf.summary.scalar('Data reward', data=total_data_reward, step=int(ep))
            
        total_reward_list.append(total_reward)
        avg_total_reward.append(sum(total_reward_list[-20:])/len(total_reward_list[-20:]))
        if "Surrogate_Accelerator" in env_id:
            PID_reward_list.append(total_pid_reward)
            avg_PID_reward.append(sum(PID_reward_list[-20:])/len(PID_reward_list[-20:]))
            data_reward_list.append(total_data_reward)

        # Logger
        logger.info('Avg Episodic Reward: {} epsilon: {}'.format(np.round(avg_total_reward[-1],3), agent.epsilon))
        if "Surrogate_Accelerator" in env_id:
            logger.info('Episodic PID Reward: %s' % str(total_pid_reward))
            logger.info('Episodic Data Reward: %s' % str(total_data_reward))

        
    
    plt.plot(avg_total_reward, linewidth=2, color="red", label="Agent Reward")
    if "Surrogate_Accelerator" in env_id:
        plt.plot(avg_PID_reward, linewidth=2, color="blue", label="PID Reward")
        plt.plot(data_reward_list, linewidth=2, color="black", label="Data Reward")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Avg Episodic Reward")
    plt.grid()
    plt.savefig("Final_Reward_Plot"+env_id+".png")
    plt.show()
    
    agent.save_model()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nepisodes", help="Number of episodes", type=int, default=100)
    parser.add_argument("--agent", help="Agent used for RL", type=str, default='KerasDQN-v0')
    parser.add_argument("--env", help="Environment used for RL", type=str, default='Surrogate_Accelerator-v1')
    parser.add_argument("--doPlay", help="Test a trained agent", type=bool, default=False)

    # Get input arguments
    args = parser.parse_args()
    nepisodes = args.nepisodes
    agent_id = args.agent
    env_id = args.env
    doPlay = args.doPlay

    # Print input settings
    run_opt(nepisodes, agent_id, env_id, doPlay)