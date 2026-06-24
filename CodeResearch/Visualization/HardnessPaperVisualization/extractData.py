import os
import re

from CodeResearch.Visualization.HardnessPaperVisualization.dataPreprocessor import preprocessDataCut, preprocessLabels, \
    getPlottingParameters, preprocessDataExpand
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays
from CodeResearch.Visualization.visualizeLearningErrors import plot_multi_errors_vs_alpha_std


def extractFiles(folder, task, mode):
    targetFolder = f'{folder}\\{task}'

    all_items = os.listdir(targetFolder)

    gradFiles = []
    noGradFiles = []

    for item in all_items:
        if not os.path.isdir(os.path.join(targetFolder, item)):
            continue

        if mode in item:
            if 'gradient' in item:
                gradientFolder = item
                gf = [(file, os.path.join(targetFolder, gradientFolder, file)) for file in
                             os.listdir(f'{targetFolder}\\{gradientFolder}') if file.endswith('txt')]
                gradFiles.extend(gf)
            else:
                noGradientFolder = item
                ngf = [(file, os.path.join(targetFolder, noGradientFolder, file)) for file in
                               os.listdir(f'{targetFolder}\\{noGradientFolder}') if file.endswith('txt')]
                noGradFiles.extend(ngf)

    return gradFiles, noGradFiles


def extract_parts(filename, modes):
    pattern = r'^\(([-]?\d+\.?\d*)\)_(\d+)_data\.txt$'
    for mode in modes:
        curModeStr = f'_{mode} '
        if curModeStr in filename:
            parts = filename.split(curModeStr, 1)
            before = parts[0]
            after = parts[1] if len(parts) > 1 else ''

            match = re.match(pattern, after)
            z = match.group(1)  # десятичное число в скобках
            n = int(match.group(2))  # натуральное число

            return before, mode, z, n

    return None, None, None, None

def fillParameters(mode, fraction, n, file):
    return {
        'number': n,
        'mode': mode,
        'fraction': fraction,
        'file': file
    }

def getKey(prefix, mode, fraction):
    return f'{prefix}___{mode}___{fraction}'

def filterFiles(files, modes):
    f = dict()

    for file, fullFile in files:
        prefix, mode, fraction, n = extract_parts(file, modes)
        if prefix is None:
            continue

        key = getKey(prefix, mode, fraction)

        if key in f:
            if f[key]['number'] < n:
                f[key] = fillParameters(mode, fraction, n, fullFile)
            continue

        f[key] = fillParameters(mode, fraction, n, fullFile)

    return f

def processFiles(grad, noGrad):
    g = filterFiles(grad, ['h&i_inc'])
    ng = filterFiles(noGrad, ['l', 'h&h_inc', 'i', 'i_cos', 'i_inner_p'])

    res = dict()
    for key, value in g.items():
        res[f'{key}_grad'] = value

    for key, value in ng.items():
        res[f'{key}_nograd'] = value

    return res

def extractConcreteTask(folder, task, fixTestMask):
    grad, noGrad = extractFiles(folder, task, fixTestMask)
    res = processFiles(grad, noGrad)
    return res

def extractTask(folder, task):
    ft = extractConcreteTask(folder, task, 'fix test')
    rs = extractConcreteTask(folder, task,'random subset')

    return {
        'fixed test': ft,
        'random subset': rs
    }

def extractFilesForParameters(r, fraction, protocol, mode=None):
    branch = r[protocol]

    resultFiles = []
    for key, value in branch.items():
        if value['fraction'] == fraction:
            if mode is None:
                resultFiles.append(value)
                continue

            if value['mode'] == mode:
                resultFiles.append(value)

    return resultFiles

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
def make_grid(image_paths, out_path, nrows=2, ncols=3, dpi=300, title=None):
    assert len(image_paths) == nrows * ncols, "Need exactly nrows*ncols images"
    fig, axes = plt.subplots(nrows, ncols, dpi=dpi)

    for ax, p in zip(axes.flat, image_paths):
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def make_2plus1_grid(image_paths, out_path, dpi=300, title=None, pad=0.02):
    """
    Layout:
      [img0 | img1]
      [   img2    ]  (spans both columns)
    """
    assert len(image_paths) == 3, "Need exactly 3 images"

    fig = plt.figure(dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], wspace=pad, hspace=pad)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])  # span both columns

    axes = [ax0, ax1, ax2]

    for ax, p in zip(axes, image_paths):
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def extractAndSave(folder, task, targetLength, fraction, protocol, startIdx=0):
    r = extractTask(folder, task.lower())
    print(r)

    files = extractFilesForParameters(r, fraction, protocol)

    labels = []
    errors = []
    maxEpochs = 0

    for file in files:
        rr = deserialize_labeles_list_of_arrays(file['file'])
        errors.append(rr[0])
        labels.append(file['mode'])
        maxEpochs = max(maxEpochs, len(rr[1]))

    errorsProcessed = preprocessDataExpand(errors, targetLength)
    labels = preprocessLabels(labels)
    title, ylabel = getPlottingParameters(task, protocol, fraction)

    xAxis = range(len(errorsProcessed[0]))

    plot_multi_errors_vs_alpha_std(errorsProcessed, xAxis, labels, task, f'{task}_{protocol}_{fraction}', len(labels),
                                   startIdx, ylabel, title)
    plot_multi_errors_vs_alpha_std(errorsProcessed, xAxis, labels, task, f'{task}_{protocol}_{fraction}_10', len(labels),
                                   10, ylabel, title)





