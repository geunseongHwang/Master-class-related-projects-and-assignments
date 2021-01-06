import os
from os import chdir
import linkpred

M = linkpred.read_network("2019_sec.graphml.graphml")
simRank = linkpred.predictors.SimRank(M,excluded=M.edges())
simRank_results = simRank.predict(c=0.4,num_iterations=1000)
top_m = simRank_results.top(30)

for authors, score in top_m.items():
    print(authors, score)
