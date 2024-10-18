#%%


#%%
import pandas as pd
import ray.data
DATASET_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
ray.data.DatasetContext.get_current().execution_options.preserve_order = True  # deterministic
ds = ray.data.read_csv(DATASET_LOC)
ds = ds.random_shuffle(seed=1234)
ds.take(1)
# %%
import sys
sys.path.append("..")
from madewithml.data import stratify_split