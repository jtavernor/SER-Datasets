import numpy as np
# min_v, max_v: min and max value present in original data
# new_min, new_max: new min and max to be min-max scaled to
# x: value to scale
def scale(min_v, max_v, new_min, new_max, x):
    return new_min + (x-min_v)*(new_max-new_min)/(max_v-min_v)

def scale_dataset(dataset, minv=1, maxv=7):
    dataset['act'] = scale(minv,maxv,-1,1,dataset['act'])
    dataset['val'] = scale(minv,maxv,-1,1,dataset['val'])
    if 'soft_act_labels' in dataset:
        dataset['soft_act_labels'] = [scale(minv, maxv, -1, 1, np.array(x)) for x in dataset['soft_act_labels']]
        # dataset['soft_act_labels'] = [scale(minv, maxv, -1, 1, x) for x in dataset['soft_act_labels']]
        dataset['soft_val_labels'] = [scale(minv, maxv, -1, 1, np.array(x)) for x in dataset['soft_val_labels']]
        # dataset['soft_val_labels'] = [scale(minv, maxv, -1, 1, x) for x in dataset['soft_val_labels']]
    return dataset