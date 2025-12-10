import numpy as np

from CodeResearch.CurriculumLearning.StatisticsProcessing.multiplePlots import plot_n_graphs
from CodeResearch.CurriculumLearning.StatisticsProcessing.statisticsExtractor import extractData
from CodeResearch.CurriculumLearning.StatisticsProcessing.visualizationHelpers import plot_with_confidence_intervals
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays

folder = '99_results'
#filePattern = 'cifar100_epoch_50_1000_0.5_1_20_8_60_43_88_NN_llmh_probs_added_x'
#filePattern = 'cifar100_epoch_50_1000_0.5_1_20_8_60_81_62_NN_llmh_probs_added_x'
#filePattern = 'cifar100_epoch_100_1000_0.5_1_20_8_40_43_88_NN_llmh_probs_added_x_product'
#filePattern = 'cifar100_epoch_100_1000_0.5_1_20_8_40_43_88_NN_llmh_probs_added_x_product_01_02'
#filePattern = 'cifar100_epoch_100_1000_0.5_1_20_8_40_47_52_NN_llmh_probs_added_x_product'
#filePattern = 'cifar100_epoch_100_1000_0.5_1_20_8_40_43_87_NN_llmh_probs_added_x_product'
#filePattern = 'cifar100_epoch_100_1000_0.5_1_20_8_40_70_91_NN_llmh_probs_added_x_product'
filePattern = 'cifar100_epoch_100_1000_0.5_1_20_8_40_9_10_NN_llmh_probs_added_x_product'

epochs = 40
iterations = 100
repeats = 20
fileNumber = 99

alphas = ['0.05', '0.1']

for alpha in alphas:

    basepostfix = f'l ({alpha})_{fileNumber}_data'
    postfixes = ['h', 'i', 'h&i']

    for p in postfixes:
        postfix = f'{p} ({alpha})_{fileNumber}_data'
        targetArray = deserialize_labeles_list_of_arrays(f'..\\{folder}\\{filePattern}_{postfix}.txt')
        baseArray = deserialize_labeles_list_of_arrays(f'..\\{folder}\\{filePattern}_{basepostfix}.txt')

        baseResults = extractData(baseArray, epochs, iterations, repeats)
        targetResults = extractData(targetArray, epochs, iterations, repeats)

        baseMean = baseResults[0]
        targetMean = targetResults[0]
        baseSigmas = baseResults[1]
        targetSigmas = targetResults[1]

        fileName = f'..\\{folder}\\{filePattern}_{postfix}.png'
        plot_with_confidence_intervals(range(epochs), baseMean, targetMean, baseSigmas, targetSigmas, [1, 2, 3], title=filePattern, labels=[f'l ({alpha})', f'{p} ({alpha})'], fileName=fileName)

        fileName = f'..\\{folder}\\{filePattern}_{postfix}_delta.png'
        plot_with_confidence_intervals(range(epochs), targetMean - baseMean, np.zeros(epochs), np.sqrt(baseSigmas**2 + targetSigmas**2), np.zeros(epochs), [1, 2, 3],
                                       title=filePattern, labels=[f'l ({alpha})', f'{p} ({alpha})'], fileName=fileName)
        l = ['e', 'r', 'i', 't']
        fileName = f'..\\{folder}\\{filePattern}_{postfix}_stds.png'
        plot_n_graphs(range(epochs), [baseResults[k] for k in range(1, 5)] + [targetResults[k] for k in range(1, 5)], labels=[f'{'l' if k < 4 else p} ({alpha}) {l[k%4]}' for k in range(8)], fileName=fileName, title=filePattern)