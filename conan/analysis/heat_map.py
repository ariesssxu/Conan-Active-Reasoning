import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
from conan.training.env import Nav_Env
from PIL import Image

env = Nav_Env(size = (6400, 6500), view=(64, 65), view_type='visual')
env.reset("get_diamond")

mat_map, _ = env.get_detailed_view()
obs = env._obs()
img = env.render()
# save img as pdf
img = Image.fromarray(img)
img.save("img.pdf")

mat_map = mat_map.T[::-1, :]
sns.heatmap(mat_map, square=True, cbar=False)
plt.xlim(0, mat_map.shape[0])
plt.ylim(0, mat_map.shape[1])
# plt.savefig("test.pdf")
# remove axis
plt.axis('off')
plt.savefig("heatmap.pdf", bbox_inches='tight', pad_inches=0)