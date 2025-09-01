import os

from CodeResearch.Visualization.visualizeAndSaveKSSI import visualizeAndSaveKSSI

nSamples = 2000
#x, y = make_random(2000)
#x, y = datasets.make_blobs(n_samples=2000, centers=2, n_features=2, random_state=42)
#x, y = make_xor(2000)
#x, y = datasets.make_circles(n_samples=nSamples, factor=0.5, noise=0.1, random_state=42)
#x, y = make_spirals(nSamples)
#x, y = load_proteins_bin("../../Data/Proteins/df_master.csv")
#x = perform_pca(x, 10)

alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
#alphas = []

taskName = "megamarket"
iterations = 200
visualizeAndSaveKSSI("..\Megamarket\Logs200", "..\Megamarket", alphas, taskName, iterations)

#taskName = "proteins_bin_kssi"
#iterations = 500
#visualizeAndSaveKSSI("..\..\PValuesFigures\PValueLogs", os.curdir, alphas, taskName, iterations)
