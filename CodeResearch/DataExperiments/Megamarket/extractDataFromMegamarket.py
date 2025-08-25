import pandas as pd
import numpy as np

category_col = "cat_level_1"
data = pd.read_parquet("../../Data/megamarket/megamarket_with_emb_not_norm.parquet")

def getNeededCounts(data, totalObjects):
    counts = data[category_col].value_counts(dropna=False)
    parts = counts / sum(counts)

    print(counts)

    neededCounts = np.floor(parts * totalObjects).astype(int)
    return neededCounts

def stratified_sample(df: pd.DataFrame, category_col: str, alloc: pd.Series, random_state: int = 42):
    parts = []
    for cat, n_take in alloc.items():
        group = df[df[category_col] == cat]
        if n_take > len(group):
            n_take = len(group)   # на всякий случай
        sample = group.sample(n=n_take, replace=False, random_state=random_state)
        parts.append(sample)
    return pd.concat(parts, ignore_index=True)

counts = getNeededCounts(data, 10000)
newData = stratified_sample(data, category_col, counts)

newData.to_parquet("../../Data/megamarket/sampled_10k.parquet", index=False, engine="pyarrow")
