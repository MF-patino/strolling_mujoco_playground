import controller.plots as plots
import pickle

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

with open("[1. 0. 0.].pkl", 'rb') as f:
    plotData = AttrDict(pickle.load(f))

plots.statisticDriftHistory(plotData)
plots.wmErrorHistory(plotData)
plots.policyEmbeddings3D(plotData)
plots.plotGaitPattern(plotData)
plots.plotLastGPSearchState(plotData)