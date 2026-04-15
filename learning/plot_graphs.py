import controller.plots as plots
from controller.plots import PLOT_DATA_DIR
import pickle, os

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

plotFiles = os.listdir(PLOT_DATA_DIR)
with open(os.path.join(PLOT_DATA_DIR, "[1. 0. 0.].pkl"), 'rb') as f:
    plotData = AttrDict(pickle.load(f))

for env_change in plotData.env_changes:
    plots.plotGaitPattern(plotData, env_change)
for gp_state in plotData.gp_states:
    plots.plotGPSearch(plotData, gp_state)
plots.statisticDriftHistory(plotData)
plots.wmErrorHistory(plotData)
plots.policyEmbeddings3D(plotData)