#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
import tensorflow as tf

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable

from agent import DQNAgent


def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)




if opt.disp_on:
    win_all = None
    win_pob = None


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# You should prepare your network training here. I suggest to put this into a
# class by itself but in general what you want to do is roughly the following
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
state_shape = int(opt.cub_siz*opt.pob_siz*opt.cub_siz*opt.pob_siz*opt.hist_len)

print("state shape {}".format(state_shape))
print("opt.act_num shape {}".format(opt.act_num))

agent = DQNAgent(state_shape, opt.act_num)
# agent.load("./save/cartpole-dqn.h5")
# batch_size = 4
batch_size = 50#32
agent.model.summary()
for layer in agent.model.layers:
    print(layer.get_output_at(0).get_shape().as_list())
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
steps = 1 * 10**6
epi_step = 0
nepisodes = 0
episode_reward = 0 #sum of a all rewards in one episode
disp_progress_n = 5 # show a full episode every n episodes
FULL_RANDOM_STEPS = 5000

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
disp_progress = False

for step in range(steps):
    if state.terminal or epi_step >= opt.early_stop or episode_reward < -6 :

        disp_progress = True if nepisodes % disp_progress_n == 0 else False


        if nepisodes % 50 == 0 and nepisodes != 0 :
            print("saved")
            agent.save("save/network.h5")
        if state.terminal:
            print("nepisodes_solved:")
        nepisodes += 1
        print("step: {}/{}, played {} episodes, episode_reward: {:.2}, epi_step {}, e: {:.2}"
                      .format(step, steps,nepisodes, episode_reward,epi_step, agent.epsilon))
        epi_step = 0
        episode_reward = 0
        agent.update_target_model()
        # reset the game
        #state = sim.newGame(opt.tgt_y, opt.tgt_x)#random agent pos
        state = sim.newGame(opt.tgt_y, opt.tgt_x, agent_fre_pos =0)#random agent pos

        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: here you would let your agent take its action
    #       remember
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # this just gets a random action for training
    if step <= FULL_RANDOM_STEPS:
        action = randrange(opt.act_num)#TODO
    else:
        action = agent.act(np.array([state_with_history.reshape(-1)]))
        #print(action)
    # action = agent.act(np.array([state_with_history.reshape(-1)]))

    # print(np.mean(state_with_history.reshape(-1)))#check if state is changing

    epi_step +=1
    next_state = sim.step(action)
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # add to the transition table
    #trans.add(state_with_history.reshape(-1), trans.one_hot_action(action), next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
    agent.remember(np.array([state_with_history.reshape(-1)]), action, next_state.reward, np.array([next_state_with_history.reshape(-1)]), next_state.terminal)
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    episode_reward += next_state.reward
    state = next_state

    # TODO every once in a while you should test your agent here so that you can track its performance
    if len(agent.memory) > batch_size and step > FULL_RANDOM_STEPS:
         e_change =True if FULL_RANDOM_STEPS >step else False
         agent.replay(batch_size, e_change)

    #if step > FULL_RANDOM_STEPS:
    #    agent.train(trans.sample_minibatch())

    if opt.disp_on and disp_progress:

        if win_all is None:
            plt.subplot(121)
            win_all = plt.imshow(state.screen)
            plt.subplot(122)
            win_pob = plt.imshow(state.pob)
        else:
            win_all.set_data(state.screen)
            win_pob.set_data(state.pob)
        plt.pause(opt.disp_interval)
        plt.draw()



# 2. perform a final test of your model and save it
# TODO
