# min_v, max_v: min and max value present in original data
# new_min, new_max: new min and max to be min-max scaled to
# x: value to scale
def scale(min_v, max_v, new_min, new_max, x):
    return new_min + (x-min_v)*(new_max-new_min)/(max_v-min_v)

def scale_dataset(item, minv=1, maxv=7):
    item['act'] = scale(minv,maxv,-1,1,item['act'])
    item['val'] = scale(minv,maxv,-1,1,item['val'])
    item['soft_act_labels'] = [scale(minv, maxv, -1, 1, x) for x in item['soft_act_labels']]
    item['soft_val_labels'] = [scale(minv, maxv, -1, 1, x) for x in item['soft_val_labels']]
    return item