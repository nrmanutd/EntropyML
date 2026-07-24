import numpy as np

def preprocessDataCut(errors):
    minimalArrays = len(errors[0])
    for err in errors:
        if len(err) < minimalArrays:
            minimalArrays = len(err)

    result = []
    for err in errors:
        result.append(err[:minimalArrays])

    return result


def preprocessDataExpand(errors, targetLength):
    maximalArrays = targetLength
    for err in errors:
        if len(err) > maximalArrays:
            maximalArrays = len(err)

    result = []
    for err in errors:
        if len(err) == maximalArrays:
            result.append(err)
            continue

        last = np.array(err[len(err) - 1: len(err) - 3: -1])
        m = np.mean(last)
        std = np.std(last)

        tempErr = []
        for i in range(maximalArrays):
            if i < len(err):
                tempErr.append(err[i])
                continue

            n = len(tempErr[-1])
            values = np.random.normal(loc=m, scale=std, size=n)
            tempErr.append(values)

        result.append(tempErr)

    return result

def preprocessLabels(labels):
    r = []
    for label in labels:
        if label == 'l':
            r.append('Rand')
        elif label == 'h&h_inc':
            r.append('Hard')
        elif label == 'h&i_inc':
            r.append('H&U')
        elif label == 'i':
            r.append('Utility')
        elif label == 'i_cos':
            r.append('Cosine')
        elif label == 'i_inner_p':
            r.append('MProb&U')
        else:
            raise ValueError(f'Unknown label: {label}')

    return r

def getPlottingParameters(task, protocol, fraction):
    latex_alpha = r'$\alpha$'
    if protocol == 'fixed test':
        title = f'{task}: Hold-out accuracy vs Epoch (Mean ± Std), {latex_alpha}={fraction}'
        return title, 'Hold-out accuracy'
    elif protocol == 'random subset':
        title = f'{task}: Cross-split accuracy vs Epoch (Mean ± Std), {latex_alpha}={fraction}'
        return title, 'Cross-split accuracy'
    else:
        raise ValueError(f'Unknown protocol: {protocol}')