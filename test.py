#%%
import ray
if ray.is_initialized():
    ray.shutdown()

ray.init()

# %%
