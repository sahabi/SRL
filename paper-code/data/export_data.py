import numpy as np
import os
import pandas as pd


t = 100
p = 100
path = ''
env = '15x9_cycling'
env = '/'+env

def averaging(a):
    a = a[:-1]
    n_samples = len(a)/100.
    print n_samples
    a = a.reshape(100,int(n_samples))
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

avg_reward_no_shield = sample(np.loadtxt(os.getcwd()+
    grid+'_0_avg_reward.data'),p=p)
ep_len_no_shield = sample(np.loadtxt(os.getcwd()+
    grid+'_0_episodelen.data'),p=p)

avg_reward_no_shield = averaging(avg_reward_no_shield)
avg_reward_no_shield = pd.DataFrame(avg_reward_no_shield)

avg_reward_no_shield_neg_reward = sample(np.loadtxt(os.getcwd()+
    grid+'_0_huge_neg_avg_reward.data'),p=p)

avg_reward_no_shield_neg_reward = averaging(avg_reward_no_shield_neg_reward)
avg_reward_no_shield_neg_reward = pd.DataFrame(avg_reward_no_shield_neg_reward)

avg_reward_shield_1_action = sample(np.loadtxt(os.getcwd()+
    grid+'_1_avg_reward.data'),p=p)
ep_len_shield_1_action = sample(np.loadtxt(os.getcwd()+
    grid+'_1_episodelen.data'),p=p)

avg_reward_shield_1_action = averaging(avg_reward_shield_1_action)
avg_reward_shield_1_action = pd.DataFrame(avg_reward_shield_1_action)

avg_reward_shield_1_action_neg_reward = sample(np.loadtxt(os.getcwd()+
    grid+'_1_neg_reward_avg_reward.data'),p=p)
ep_len_shield_1_action_neg_reward = sample(np.loadtxt(os.getcwd()+
    grid+'_1_neg_reward_episodelen.data'),p=p)

avg_reward_shield_1_action_neg_reward = averaging(avg_reward_shield_1_action_neg_reward)
avg_reward_shield_1_action_neg_reward = pd.DataFrame(avg_reward_shield_1_action_neg_reward)

avg_reward_shield_3_action = sample(np.loadtxt(os.getcwd()+
    grid+'_3_avg_reward.data'),p=p)
ep_len_shield_3_action = sample(np.loadtxt(os.getcwd()+
    grid+'_3_episodelen.data'),p=p)

avg_reward_shield_3_action = averaging(avg_reward_shield_3_action)
avg_reward_shield_3_action = pd.DataFrame(avg_reward_shield_3_action)

avg_reward_shield_3_action_neg_reward = sample(np.loadtxt(os.getcwd()+
    grid+'_3_neg_reward_avg_reward.data'),p=p)
ep_len_shield_3_action_neg_reward = sample(np.loadtxt(os.getcwd()+
    grid+'_3_neg_reward_episodelen.data'),p=p)

avg_reward_shield_3_action_neg_reward = averaging(avg_reward_shield_3_action_neg_reward)
avg_reward_shield_3_action_neg_reward = pd.DataFrame(avg_reward_shield_3_action_neg_reward)

result = pd.concat([avg_reward_no_shield, avg_reward_no_shield_neg_reward, avg_reward_shield_1_action, 
    avg_reward_shield_1_action_neg_reward, avg_reward_shield_3_action, avg_reward_shield_3_action_neg_reward], axis=1)
result.to_csv(grid[1:]+'.dat',sep=' ', index_label='x', header=['no_shield','shield_1','shield_1_neg','shield_3','shield_3_neg'])
