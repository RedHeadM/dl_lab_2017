import numpy as np
import matplotlib.pyplot as plt

from utils import Options
from test_agent import test_model
from train_agent import train_model

from keras import backend as K

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
    opt.disp_on = False
    hist_len = np.arange(start=1, stop=11)
    results_success_rate = []
    results_astar_diff = []
    with K.get_session():
        for l in hist_len:
            opt.hist_len = l
            print("start with opt.hist_len: {}".format(opt.hist_len))
            train_model(opt, mdl_name,epochs=5)
            [success_rate, astar_diff] = test_model(opt,mdl_name)
            results_success_rate.append(success_rate)
            results_astar_diff.append(astar_diff)
    # Four polar axes
    f, axarr = plt.subplots(1,2)
    print(hist_len)
    axarr[0].scatter(hist_len, results_success_rate)
    # axarr[0].set_title('results_success_rate')
    axarr[1].set_ylabel(r'success rate',usetex=True)
    axarr[1].scatter(hist_len, results_astar_diff)
    # axarr[1].set_title('differnce to astar in number of steps')
    axarr[1].set_ylabel(r'num of steps',usetex=True)
    axarr[1].set_xlabel(r'history length',usetex=True)

    # Fine-tune figure; make subplots farther from each other.
    f.subplots_adjust(hspace=0.3)
    helper_save(plt_file_name)


def plt_pob_siz_results(plt_file_name = None):
    mdl_name = "list_len_mdl.h5"
    opt = Options()
    opt.disp_on = False
    pob_siz_len = np.arange(start=3, stop=11)
    results_success_rate = []
    results_astar_diff = []

    with K.get_session():
        for p in pob_siz_len:
            opt.pob_siz = p
            print("start with opt.pob_siz: {}".format(opt.pob_siz))
            train_model(opt, mdl_name, epochs=10)
            [success_rate, astar_diff] = test_model(opt,mdl_name)
            results_success_rate.append(success_rate)
            results_astar_diff.append(astar_diff)
    # Four polar axes
    f, axarr = plt.subplots(1,2)
    print(hist_len)
    axarr[0].scatter(hist_len, results_success_rate)
    # axarr[0].set_title('results_success_rate')
    axarr[1].set_ylabel(r'success rate',usetex=True)
    axarr[1].scatter(hist_len, results_astar_diff)
    # axarr[1].set_title('differnce to astar in number of steps')
    axarr[1].set_ylabel(r'num of steps',usetex=True)
    axarr[1].set_xlabel(r'history length',usetex=True)

    # Fine-tune figure; make subplots farther from each other.
    f.subplots_adjust(hspace=0.3)


if __name__ == "__main__":
    # plt_hist_len_results()
    plt_hist_len_results("hist_len")
