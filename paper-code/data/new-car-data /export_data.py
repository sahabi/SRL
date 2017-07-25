import numpy as np
import os
import pandas as pd
t = 1
p = 100
grid = 'car'
grid = '/'+grid

def averaging(a):
    print len(a)
    a = a[:-1]
    n_samples = len(a)/1
    print n_samples
    if len(a)%5. == 0:
        a = a.reshape(1,int(n_samples))
    else:
        print len(a)%100.
    return a.mean(axis=1)

def window(size):
    return np.ones(size)/float(size)

def sample(data,n=None,p=None):
    if n is None:
        if p is None:
            return data
        else:
            return data[:int(len(data)/(100./p))]
    else:
        return data[:n]

avg_reward_no_shield =sample(np.loadtxt(os.getcwd()+
    grid+'-noshield.csv'),p=p)


# avg_reward_no_shield = averaging(avg_reward_no_shield)
avg_reward_no_shield = pd.DataFrame(avg_reward_no_shield)

avg_reward_shield_1_action = sample(np.loadtxt(os.getcwd()+
    grid+'-shielded.csv'),p=p)


# avg_reward_shield_1_action = averaging(avg_reward_shield_1_action)
avg_reward_shield_1_action = pd.DataFrame(avg_reward_shield_1_action)


avg_reward_shield_pre_1_action = sample(np.loadtxt(os.getcwd()+
    grid+'-shielded-pre.csv'),p=p)


# avg_reward_shield_1_action = averaging(avg_reward_shield_1_action)
avg_reward_shield_pre_1_action = pd.DataFrame(avg_reward_shield_pre_1_action)

result = pd.concat([avg_reward_no_shield, avg_reward_shield_1_action, avg_reward_shield_pre_1_action], axis=1)
result.to_csv(grid[1:]+'.dat',sep=' ', index_label='x', header=['no_shield','shield_1','shield_1_pre'])