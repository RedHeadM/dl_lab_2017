import numpy as np

class Options:
    #
    disp_on = False # you might want to set it to False for speed
    map_ind = 1
    change_tgt = False #rand goal pos if true
    states_fil = "states.csv"
    labels_fil = "labels.csv"
    network_fil = "network.json"
    weights_fil = "network.h5"
    # simulator config
    disp_interval = .005
    cub_siz = 5
    pob_siz = 5     # for partial observation, must be not even
    # run get_data if cub_siz or pob_siz is changed
    if map_ind == 0:

        # this defines the goal positionw
        tgt_y = 12 #12
        tgt_x = 11 #11
        early_stop = 50
    elif map_ind == 1:
        # this defines the goal positionw
        tgt_y = 5
        tgt_x = 5
        early_stop = 75
    assert not pob_siz %2 ==0,"pob_siz must be not even"
    state_siz = (pob_siz * cub_siz) ** 2 # when use pob as input
    if change_tgt:
        tgt_y = None
        tgt_x = None
    act_num = 5
    # [3,5,7,11,13]
    # traing hyper params
    hist_len = 3
    minibatch_size  = 32
    n_minibatches   = 500
    valid_size      = 500
    eval_nepisodes  = 50

    data_steps  = n_minibatches * minibatch_size + valid_size
    eval_steps  = early_stop * eval_nepisodes
    assert eval_steps > early_stop
    eval_freq   = n_minibatches # evaluate after each epoch
    prog_freq   = 500

class State: # return tuples made easy
    def __init__(self, action, reward, screen, terminal, pob):
        self.action   = action
        self.reward   = reward
        self.screen   = screen
        self.terminal = terminal
        self.pob      = pob


# The following functions were taken from scikit-image
# https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py

def rgb2gray(rgb):
    if rgb.ndim == 2:
        return np.ascontiguousarray(rgb)

    gray = 0.2125 * rgb[..., 0]
    gray[:] += 0.7154 * rgb[..., 1]
    gray[:] += 0.0721 * rgb[..., 2]

    return gray
