import os
'''
Run from parent directory of LOGS
'''

cluster_logs_flag = True
logs = 'CLUSTER_LOGS' if cluster_logs_flag else 'LOGS'
data_names = [f for f in os.listdir(logs) if 'profile_M' in f \
        or '.data' in f or '.index' in f or 'table.mod' in f]
subdir_names = [f[0] for f in os.walk(logs) if logs+'/' in f[0]]

for d in data_names:
    if 'profile_M' in d and '.data' in d:
        to_subdir = d.split('_')[1]+'_'+d.split('_')[2]
    if 'history.data' in d:
        to_subdir = d.split('_')[0]+'_'+d.split('_')[1]
    os.rename(logs+'/'+d, logs+'/'+to_subdir+'/'+d)
