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
import argparse
from agent import QMazeAgent

import sys
sys.path.append('..')
#import simulation for ex03 for astar
from exercise3robotics_comp.simulator import Simulator as SimulatorAstar
from exercise3robotics_comp.test_agent import get_astar_steps

from utils import append_to_hist
parser = argparse.ArgumentParser()




N_EPISODES_TOTAL_TEST = 1000 #number of total trainign game episodes
DISP_PROGRESS_AFTER_N_EPISODES = 1 # show a full episode every n episodes for opt.disp_on is true


def helper_save(plt_file_name):
    if plt_file_name is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(plt_file_name+'.pdf', format='pdf', dpi=1000)#save as pdf first
        from matplotlib2tikz import save as tikz_save
        # tikz_save('../report/ex1/plots/test.tex', figureheight='4cm', figurewidth='6cm')
        tikz_save(plt_file_name + ".tikz", figurewidth="\\matplotlibTotikzfigurewidth", figureheight="\\matplotlibTotikzfigureheight",strict=False)

def start_new_game():
    """start a new game return astar steps need"""
    # reset the game
    simAStar.newGame(opt.tgt_y, opt.tgt_x)#random agent pos
    state = sim.newGame(opt.tgt_y, opt.tgt_x, agent_fre_pos =0)#agent fixed start pos
    #same start pos
    sim.obj_pos[sim.bot_ind][0] = simAStar.obj_pos[simAStar.bot_ind][0]
    sim.obj_pos[sim.bot_ind][1] = simAStar.obj_pos[simAStar.bot_ind][1]
    return get_astar_steps(simAStar),state


# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
simAStar = SimulatorAstar(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)





parser.add_argument("-w", "--weights", help=".h5 weights_file_name for conv network",
                    default=opt.weights_fil)
args = parser.parse_args()
weights_file_name = args.weights


if opt.disp_on:
    win_all = None
    win_pob = None


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#Agent
input_shape_dense = int(opt.cub_siz*opt.pob_siz*opt.cub_siz*opt.pob_siz*opt.hist_len)
input_shape_conv = (opt.cub_siz*opt.pob_siz,opt.cub_siz*opt.pob_siz,opt.hist_len)
use_conv = True
agent = QMazeAgent(input_shape_conv, opt.act_num,use_conv=use_conv)
agent.model.summary()#print mdl
agent.load(weights_file_name)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#Param for training loop
episode_reward_hist = []# history of total episode rewards
epi_step_hist = [] #history of total episode step needed to solve or ealy step
epi_step = 0
nepisodes = 0
episode_reward = 0 #sum of a all rewards in one episode
disp_progress = False
astar_steps,state =start_new_game()
astar_steps_hist = [astar_steps]
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
nepisodes_solved_cnt = 0
termin_state = []

while nepisodes < N_EPISODES_TOTAL_TEST:
    if state.terminal or epi_step >= opt.early_stop or episode_reward < -10 :

        disp_progress = True if nepisodes % DISP_PROGRESS_AFTER_N_EPISODES == 0 else False
        nepisodes += 1


        if state.terminal:
            nepisodes_solved_cnt +=1
            print("nepisodes_solved:")

        print("played {}/{} episodes, episode_reward: {:.2}, epi_step {}, e: {:.2}"
                      .format(nepisodes,N_EPISODES_TOTAL_TEST, episode_reward,epi_step, agent.epsilon))
        termin_state.append(state.terminal)
        epi_step_hist.append(epi_step)
        episode_reward_hist.append(episode_reward)
        episode_reward = 0
        epi_step = 0
        astar_steps,state =start_new_game()
        astar_steps_hist.append(astar_steps)

        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)

    if use_conv:
        action = agent.act(np.array([state_with_history]),False)
    else:
        action = agent.act(np.array([state_with_history.reshape(-1)]),False)

    epi_step +=1
    next_state = sim.step(action)
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))

    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    episode_reward += next_state.reward
    state = next_state

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

epi_step_hist =np.array(epi_step_hist)
astar_steps_hist =np.array(astar_steps_hist)
astar_steps_hist =astar_steps_hist[0:-1]#rm last
astar_steps_hist[astar_steps_hist == None] = 0
#only compare if episode was successful
astar_steps_hist = astar_steps_hist[termin_state]
epi_step_hist = epi_step_hist[termin_state]
diff_to_astar = epi_step_hist-astar_steps_hist
print("==============================")
print("success rate {}".format(nepisodes_solved_cnt/N_EPISODES_TOTAL_TEST))
print("mean diff to astare if successful {}".format(np.mean(diff_to_astar)))

plot_diff_astar_eps = np.arange(N_EPISODES_TOTAL_TEST)
plot_diff_astar_eps = plot_diff_astar_eps[termin_state]#only succ. runs


f, axarr = plt.subplots(2,1)

axarr[0].plot(episode_reward_hist)
axarr[0].set_ylabel(r'Total Reward',usetex=True)

axarr[1].plot(plot_diff_astar_eps,diff_to_astar)
axarr[1].set_ylabel(r'Difference to astar',usetex=True)

axarr[0].set_xlabel(r'Episode',usetex=True)
axarr[1].set_xlabel(r'Episode',usetex=True)

helper_save("plots/eval_test")
# 2. perform a final test of your model and save it
