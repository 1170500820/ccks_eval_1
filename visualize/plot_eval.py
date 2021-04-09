import matplotlib.pyplot as plt
import numpy as np
import pickle
import numpy as np

data = pickle.load(open('../train/eval_map_6.pk', 'rb'))
data_np = np.array(data)

Z = np.random.rand(6, 10)

fig, ax = plt.subplots(1, 1)

c = ax.pcolor(data_np)
ax.set_title('default')


fig.tight_layout()
plt.show()
