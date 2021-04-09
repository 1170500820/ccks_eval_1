import pickle
import numpy as np

data = pickle.load(open('../train/single_tem/eval_map_6.pk', 'rb'))
val_sentences, val_types, spans, val_syntactic_features = \
        pickle.load(open('../val_data_for_trigger_extraction.pk', 'rb'))
a = np.array(data)

a_mean = a.mean(0)

type_means = {}
for i, m in enumerate(a_mean):
    t = val_types[i]
    if t in type_means:
        type_means[t].append(m)
    else:
        type_means[t] = [m]

for key, value in type_means.items():
    tm = np.mean(value)
    print(key, tm)