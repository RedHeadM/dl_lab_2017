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
batch_size = 32
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
disp_progress_n = 20 # show a full episode every n episodes

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
disp_progress = False
for step in range(steps):
    if state.terminal or epi_step >= opt.early_stop:

        disp_progress = True if nepisodes % disp_progress_n == 0 else False

        if state.terminal:
            print("nepisodes_solved")
        nepisodes += 1
        print("step: {}/{}, played {} episodes, episode_reward: {:.2}, e: {:.2}"
                      .format(step, steps,nepisodes, episode_reward, agent.epsilon))
        epi_step = 0
        episode_reward = 0

        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)

        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: here you would let your agent take its action
    #       remember
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # this just gets a random action
    # action = randrange(opt.act_num)#TODO
    action = agent.act(np.array([next_state_with_history.reshape(-1)]))
    epi_step +=1

    action_onehot = trans.one_hot_action(action)
    next_state = sim.step(action)
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # add to the transition table
    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
    agent.remember(np.array([state_with_history.reshape(-1)]), action, next_state.reward,np.array([next_state_with_history.reshape(-1)]), next_state.terminal)
    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    episode_reward += next_state.reward

    state = next_state
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: here you would train your agent
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
    # print("shape state_batch {}".format(state_batch.shape))
    # print("shape state_with_history.reshape(-1) {}".format(state_with_history.reshape(-1).shape))
    # TODO train me here
    # this should proceed as follows:
    # 1) pre-define variables and networks as outlined above
    # 1) here: calculate best action for next_state_batch
    # TODO:
    # action_batch_next = CALCULATE_ME
    # 2) with that action make an update to the q values
    #    as an example this is how you could print the loss
    #print(sess.run(loss, feed_dict = {x : state_batch, u : action_batch, ustar : action_batch_next, xn : next_state_batch, r : reward_batch, term : terminal_batch}))


    # TODO every once in a while you should test your agent here so that you can track its performance
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

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
