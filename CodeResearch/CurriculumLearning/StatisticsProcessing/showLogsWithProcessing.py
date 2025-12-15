from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays
from CodeResearch.Visualization.visualizeLearningErrors import plot_multi_errors_vs_alpha_std

betas = [0.05, 0.1, 0.2, 0.5, 1]
epochs = 60
counter = 19
#prefix = 'cifar100_epoch_20_100_0.5_1_20_20_60_43_87_NN_all_classes_60_epoch_xy_prod'
prefix = 'cifar100_epoch_20_100_0.5_1_20_20_60_43_88_NN_all_classes_60_epoch_xy_prod'
#prefix = 'cifar100_epoch_20_100_0.5_1_20_20_60_47_52_NN_all_classes_60_epoch_xy_prod'
#prefix = 'cifar100_epoch_20_100_0.5_1_20_20_60_5_6_NN_all_classes_60_epoch'
#prefix = 'cifar100_epoch_20_100_0.5_1_20_20_60_47_52_NN_all_classes_60_epoch'
#prefix = 'cifar100_epoch_20_100_0.5_1_20_20_60_43_87_NN_all_classes_60_epoch'
#prefix = 'cifar100_epoch_20_100_0.5_1_20_20_60_43_88_NN_all_classes_60_epoch'
folder = '..\\results_1512\\cifar100_epoch'
xAxis = range(epochs)
baseLabels = ['l', 'i', 'h', 'h&i']
labels = []
for l in baseLabels:
    labels = labels + [f'{l} ({beta})' for beta in betas]

errors = []
for i in range(len(labels)):
    fileName = f'{folder}\\{prefix}_{labels[i]}_{counter}_data.txt'

    r = deserialize_labeles_list_of_arrays(fileName)
    errors.append(r[0])


plot_multi_errors_vs_alpha_std(errors, xAxis, labels, f'{folder}\\Processed', f'{prefix}_{counter}_processed', len(betas))
plot_multi_errors_vs_alpha_std(errors, xAxis, labels, f'{folder}\\Processed', f'{prefix}_5_{counter}_processed', len(betas), 5)
plot_multi_errors_vs_alpha_std(errors, xAxis, labels, f'{folder}\\Processed', f'{prefix}_20_{counter}_processed', len(betas), 20)