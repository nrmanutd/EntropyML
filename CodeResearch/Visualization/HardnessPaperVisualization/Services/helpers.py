def getDatasetVisualizationName(task):
    if task == 'mnist':
        return 'MNIST'
    elif task == 'cifar':
        return 'CIFAR 10'
    elif task == 'cifar100':
        return 'CIFAR 100'
    elif task == 'svhn':
        return 'SVHN'
    else:
        raise ValueError(f'Incorrect dataset name: {task}')

def getMethodVisualizationName(method):
    if method == 'rand':
        return 'Random'
    elif method == 'EL2N':
        return 'EL2N'
    elif method == 'GradNorm':
        return 'GraND'
    elif method == 'chg_inc':
        return 'CHG'
    elif method == 'forgetting':
        return 'Forgetting'
    elif method == 'k-centered_inc':
        return '$k$-center greedy'
    elif method == 'boss':
        return 'BOSS'
    elif method == 'GradNorm_inc':
        return '$\\boldsymbol{\\mathrm{GraND}_{\\mathrm{inc}}}$'
    elif method == 'h':
        return 'h'
    elif method == 'hGradNorm':
        return '$h \\times GraND$'
    elif method == 'h_inc':
        return '$\\boldsymbol{h_{\\mathrm{inc}}}$'
    elif method == 'h&GradNorm_inc':
        return 'H&GradNorm (Ours)'
    else:
        raise ValueError(f'Incorrect method: {method}')