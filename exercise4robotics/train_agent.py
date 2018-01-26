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

from utils import append_to_hist

def helper_save(plt_file_name):
    if plt_file_name is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(plt_file_name+'.pdf', format='pdf', dpi=1000)#save as pdf first
        from matplotlib2tikz import save as tikz_save
        # tikz_save('../report/ex1/plots/test.tex', figureheight='4cm', figurewidth='6cm')
        tikz_save(plt_file_name + ".tikz", figurewidth="\\matplotlibTotikzfigurewidth", figureheight="\\matplotlibTotikzfigureheight",strict=False)




N_EPISODES_TOTAL_TRAIN = 700 #number of total trainign game episodes
SAVE_AFTER_N_EPISODES = 50
DISP_PROGRESS_AFTER_N_EPISODES = 5 # show a full episode every n episodes for opt.disp_on is true
FULL_RANDOM_EPISODES = 5#two full random episodes before training


# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 10
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)




if opt.disp_on:
    win_all = None
    win_pob = None


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# You should prepare your network training here. I suggest to put this into a
input_shape_dense = int(opt.cub_siz*opt.pob_siz*opt.cub_siz*opt.pob_siz*opt.hist_len)
input_shape_conv = (opt.cub_siz*opt.pob_siz,opt.cub_siz*opt.pob_siz,opt.hist_len)

use_conv =True
agent = DQNAgent(input_shape_conv, opt.act_num,use_conv=use_conv)
agent.model.summary()#print mdl
agent.load('plots/rand_start_hist_4.h5')
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#Param for training loop
episode_reward_hist = []# history of total episode rewards
epi_step_hist = [] #history of total episode step needed to solve or ealy step
epi_step = 0
nepisodes = 0
episode_reward = 0 #sum of a all rewards in one episode
disp_progress = False
state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)


while nepisodes < N_EPISODES_TOTAL_TRAIN:
    if state.terminal or epi_step >= opt.early_stop or episode_reward < -10 :

        disp_progress = True if nepisodes % DISP_PROGRESS_AFTER_N_EPISODES == 0 else False
        nepisodes += 1

        if nepisodes % SAVE_AFTER_N_EPISODES == 0 and nepisodes != 0 :
            print("saved")
            agent.save("save/" + opt.weights_fil)

        if state.terminal:
            print("nepisodes_solved:")

        print("played {}/{} episodes, episode_reward: {:.2}, epi_step {}, e: {:.2}"
                      .format(nepisodes,N_EPISODES_TOTAL_TRAIN, episode_reward,epi_step, agent.epsilon))

        epi_step_hist.append(epi_step)
        episode_reward_hist.append(episode_reward)
        episode_reward = 0
        epi_step = 0
        agent.update_target_model()
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)#random agent pos
        # state = sim.newGame(opt.tgt_y, opt.tgt_x, agent_fre_pos =0)#agent fixed start pos

        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)

    if nepisodes <= FULL_RANDOM_EPISODES:
        action = randrange(opt.act_num)#TODO
    else:
        if use_conv:
            action = agent.act(np.array([state_with_history]))
        else:
            action = agent.act(np.array([state_with_history.reshape(-1)]))

    epi_step +=1
    next_state = sim.step(action)
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    # add to the transition table
    trans.add(state_with_history.reshape(-1), trans.one_hot_action(action), next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)

    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    episode_reward += next_state.reward
    state = next_state

    if nepisodes > FULL_RANDOM_EPISODES:
        agent.train(trans.sample_minibatch())

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

#plts rewards and step per episode

f, axarr = plt.subplots(2,1)

axarr[0].plot(episode_reward_hist)
axarr[0].set_ylabel(r'Total Reward',usetex=True)

axarr[1].plot(epi_step_hist)
axarr[1].set_ylabel(r'Number of steps',usetex=True)

axarr[0].set_xlabel(r'Episode',usetex=True)
axarr[1].set_xlabel(r'Episode',usetex=True)

helper_save("plots/test_rand_goal_pos")
# 2. perform a final test of your model and save it
