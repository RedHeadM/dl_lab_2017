import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from random import randrange
# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
import copy
from keras.models import load_model
import time
from transitionTable import TransitionTable
import random

def get_astar_steps(sim):
    """return number of optimal steps """
    epi_step = 0
    while not sim.state_terminal:
        state = sim.step()# will perform A* actions
        epi_step += 1
        if sim.state_terminal:
            return epi_step

def test_model(opt = Options(),mdl_load_name='my_model.h5'):
    """validation to astar
        return [success_rate, astar_diff]
    """
    # 0. initialization

    sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
    print(opt.state_siz)
    trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                                 opt.minibatch_size, opt.valid_size,
                                 opt.states_fil, opt.labels_fil)
    state_length = (opt.cub_siz*opt.pob_siz)
    state_history = np.zeros((1,state_length,state_length,opt.hist_len))
    #load traind mdl
    model = load_model(mdl_load_name)

    # 1. control loop
    if opt.disp_on:
        win_all = None
        win_pob = None
    epi_step = 0    # #steps in current episode
    nepisodes = 0   # total #episodes executed
    nepisodes_solved = 0
    action = 0     # action to take given by the network

    # start a new game
    state = sim.newGame(opt.tgt_y, opt.tgt_x)
    astar_num_steps = get_astar_steps(copy.deepcopy(sim))

    astar_num_steps_arr = []
    agent_num_steps_arr = []

    for step in range(opt.eval_steps):

        # check if episode ended
        if state.terminal or epi_step >= opt.early_stop:
            if state.terminal:
                nepisodes_solved += 1
            # print("astar_num_steps: {} agent steps: {} ".format(astar_num_steps,epi_step))
            astar_num_steps_arr.append(astar_num_steps)
            agent_num_steps_arr.append(epi_step)
            nepisodes += 1
            # start a new game
            state = sim.newGame(opt.tgt_y, opt.tgt_x)
            astar_num_steps = get_astar_steps(copy.deepcopy(sim))

            epi_step = 0
        else:
            #   here you would let your agent take its action
            gray_state = rgb2gray(state.pob)
            gray_state = gray_state.reshape(1,opt.state_siz)
            trans.add_recent(step, gray_state)
            recent = trans.get_recent()
            recent_shaped = recent.reshape(1,state_length,state_length,opt.hist_len)
            action = np.argmax(model.predict(recent_shaped))
            state = sim.step(action)

            epi_step += 1

        if step % opt.prog_freq == 0:
            print("step {}".format(step))

        if opt.disp_on:
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

    # 2. calculate statistics
    success_rate = float(nepisodes_solved) / float(nepisodes)
    print("this session was: {}".format(success_rate))
    # 3. additional analysis

    agent_num_steps_arr=np.array(agent_num_steps_arr)
    astar_num_steps_arr=np.array(astar_num_steps_arr)
    astar_num_steps_arr[astar_num_steps_arr == None] = 0 #set to zero if start was on goal
    #only compute mead diff to astare where goal found

    print("sahpe form ",astar_num_steps_arr.shape)
    astar_num_steps_arr = astar_num_steps_arr[agent_num_steps_arr< opt.early_stop]
    print("sahpe to",astar_num_steps_arr.shape)
    #change after astar_num_steps_arr
    agent_num_steps_arr = agent_num_steps_arr[agent_num_steps_arr< opt.early_stop]
    astar_diff = np.mean(agent_num_steps_arr-astar_num_steps_arr)
    print("avg diff to astar: {}".format(astar_diff))
    return [success_rate, astar_diff]

if __name__ == "__main__":
    test_model()
