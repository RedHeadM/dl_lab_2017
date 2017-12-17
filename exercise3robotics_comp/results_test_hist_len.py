import numpy as np
import matplotlib.pyplot as plt

from utils import Options
from test_agent import test_model
from train_agent import train_model
from get_data import get_data

from keras import backend as K

EPOCHS = 5

def helper_save(plt_file_name):
    if plt_file_name is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(plt_file_name+'.pdf', format='pdf', dpi=1000)
        from matplotlib2tikz import save as tikz_save
        # tikz_save('../report/ex1/plots/test.tex', figureheight='4cm', figurewidth='6cm')
        tikz_save(plt_file_name + ".tikz", figurewidth="\\matplotlibTotikzfigurewidth", figureheight="\\matplotlibTotikzfigureheight",strict=False)

def plt_hist_len_results(plt_file_name = None):
    mdl_name = "list_len_mdl.h5"
    opt = Options()
    hist_len = np.arange(start=1, stop=10)
    results_success_rate_map_zero = []
    results_astar_diff = []
    map_ind = opt.map_ind
    opt.map_ind =0
    get_data(opt)#generaten new data with the map 0
    opt.map_ind = map_ind
    print("map used {}".format(opt.map_ind))
    with K.get_session():
        for l in hist_len:
            opt.hist_len = l
            print("start with opt.hist_len: {}".format(opt.hist_len))
            train_model(opt, mdl_name,epochs=EPOCHS)
            [success_rate, astar_diff] = test_model(opt,mdl_name)
            results_success_rate_map_zero.append(success_rate)
            results_astar_diff.append(astar_diff)
    # Four polar axes
    f, axarr = plt.subplots(1,2)
    axarr[0].scatter(hist_len, results_success_rate_map_zero)
    # axarr[0].set_title('map 1')
    axarr[0].set_ylabel(r'success rate',usetex=True)

    axarr[1].scatter(hist_len, results_astar_diff)
    # axarr[1].set_title('map 1')
    axarr[1].set_ylabel(r'mean differnce to astar in steps',usetex=True)

    axarr[1].set_xlabel(r'history length',usetex=True)
    axarr[0].set_xlabel(r'history length',usetex=True)

    # Fine-tune figure; make subplots farther from each other.
    f.subplots_adjust(hspace=0.3)
    helper_save(plt_file_name)


def plt_pob_siz_results(plt_file_name = None):
    mdl_name = "list_len_mdl.h5"
    opt = Options()
    opt.disp_on = False
    pob_siz_len = np.arange(start=3, stop=11,step=2) #not even steps needed
    # pob_siz_len =[5,3,5,9,1]
    results_success_rate_map_zero = []
    results_astar_diff = []

    with K.get_session():
        for p in pob_siz_len:
            opt.pob_siz = p
            opt.state_siz = (p * opt.cub_siz) ** 2
            print("start with opt.pob_siz: {}".format(opt.pob_siz))
            print("start with opt.state_siz: {}".format(opt.state_siz))
            get_data(opt)#generaten new data
            train_model(opt, mdl_name, epochs=EPOCHS)
            [success_rate, astar_diff] = test_model(opt,mdl_name)
            results_success_rate_map_zero.append(success_rate)
            results_astar_diff.append(astar_diff)

    # Four polar axes
    f, axarr = plt.subplots(1,2)
    axarr[0].scatter(pob_siz_len, results_success_rate_map_zero)
    axarr[0].set_ylabel(r'success rate',usetex=True)
    axarr[1].scatter(pob_siz_len, results_astar_diff)
    # axarr[1].set_title('differnce to astar in number of steps')
    axarr[1].set_ylabel(r'mean differnce to astar in steps',usetex=True)

    axarr[0].set_xlabel(r'view size',usetex=True)
    axarr[1].set_xlabel(r'view size',usetex=True)
    helper_save(plt_file_name)


if __name__ == "__main__":
    #change opt.change_tgt = True #rand goal pos if true
    #change opt.map_ind = 1/0 for the differnt map
    plt_hist_len_results("hist_len_fixed_goal_map_0_pob_siz_5")
    # plt_pob_siz_results("pob_siz")
