import math
import pandas as pd

from CodeResearch.calculateAndVisualizeEmpiricalDistribution import calculateAndVisualizeEmpiricalDistribution
from calcEntropy import calculateAndVisualizeSeveralEntropies
from ucimlrepo import fetch_ucirepo
import numpy as np

#showRandom(150, 2)
#showCircles(150, 2)
#showCircles(1500, 2)
#showCircles(154, 4)
#showCircles(1504, 4)
#showTaskFromUciById(53) #iris
#showTaskFromUciById(186) #wine quality
#showTaskFromUciById(17) #breast cancer wisconsin
#showTaskFromUciById(602) #dry bean
#showTaskFromUciById(54) #isolet


def empiricalDistributionById(id):
    set = fetch_ucirepo(id=id)

    dataSet = np.array(set.data.features)
    target = np.array(set.data.targets)

    calculateAndVisualizeEmpiricalDistribution(dataSet, target, set.metadata.name)

    return

empiricalDistributionById(53)