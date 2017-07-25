import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pylab
from pylab import arange,pi,sin,cos,sqrt
import pandas as pd

#!python numbers=disable
fig_width_pt = 206.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_size = [fig_width,fig_height]
print fig_size
params = {'backend': 'ps',
          'axes.labelsize': 12,
          'text.fontsize': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'text.usetex': True,
          'figure.figsize': fig_size}
pylab.rcParams.update(params)
t = 100
p = 50
grid = '9x9_illustrative'
#grid = '15x9_cycling'
grid = '/'+grid
# os.chdir('wsarsa')
title = '9x9 Grid-world'
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

def averaging(a):
    # a = a[:-1]
    n_samples = len(a)/100.
    print n_samples
    a = a.reshape(100,int(n_samples))
    return a.mean(axis=1)

#Q

avg_reward_no_shield = sample(np.loadtxt(os.getcwd()+
    grid+'_0_avg_reward.data'),p=p)
plt.plot(np.convolve(avg_reward_no_shield,window(t),'same')[10:-80], label='no shield Q',c='red')

avg_reward_no_shield = averaging(avg_reward_no_shield)
avg_reward_no_shield = pd.DataFrame(avg_reward_no_shield)
avg_reward_no_shield.to_csv(grid[1:]+'_0_avg_reward.dat',sep=' ', index_label='x', header=['y'])

avg_reward_no_shield_neg_reward = sample(np.loadtxt(os.getcwd()+
    grid+'_0_huge_neg_avg_reward.data'),p=p)
plt.plot(np.convolve(avg_reward_no_shield_neg_reward,window(t),'same')[10:-80], label='no shield Q Neg Reward', c='blue')

avg_reward_no_shield_neg_reward = averaging(avg_reward_no_shield_neg_reward)
avg_reward_no_shield_neg_reward = pd.DataFrame(avg_reward_no_shield_neg_reward)
avg_reward_no_shield_neg_reward.to_csv(grid[1:]+'_0_huge_neg_avg_reward.dat',sep=' ', index_label='x', header=['y'])

avg_reward_shield_1_action = sample(np.loadtxt(os.getcwd()+
    grid+'_1_avg_reward.data'),p=p)
plt.plot(np.convolve(avg_reward_shield_1_action,window(t),'same')[10:-80], label='shield-1 action Q', c='orange')

avg_reward_shield_1_action = averaging(avg_reward_shield_1_action)
avg_reward_shield_1_action = pd.DataFrame(avg_reward_shield_1_action)
avg_reward_shield_1_action.to_csv(grid[1:]+'_1_avg_reward.dat',sep=' ', index_label='x', header=['y'])

avg_reward_shield_1_action_neg_reward = sample(np.loadtxt(os.getcwd()+
    grid+'_1_neg_reward_avg_reward.data'),p=p)
plt.plot(np.convolve(avg_reward_shield_1_action_neg_reward,window(t),'same')[10:-30], label='shield-1 action-neg reward Q', c='green')

avg_reward_shield_1_action_neg_reward = averaging(avg_reward_shield_1_action_neg_reward)
avg_reward_shield_1_action_neg_reward = pd.DataFrame(avg_reward_shield_1_action_neg_reward)
avg_reward_shield_1_action_neg_reward.to_csv(grid[1:]+'_1_neg_reward_avg_reward.dat',sep=' ', index_label='x', header=['y'])

avg_reward_shield_3_action = sample(np.loadtxt(os.getcwd()+
    grid+'_3_avg_reward.data'),p=p)
plt.plot(np.convolve(avg_reward_shield_3_action,window(t),'same')[10:-80], label='shield-3 action Q', c='gray')

avg_reward_shield_3_action = averaging(avg_reward_shield_3_action)
avg_reward_shield_3_action = pd.DataFrame(avg_reward_shield_3_action)
avg_reward_shield_3_action.to_csv(grid[1:]+'_3_avg_reward.dat',sep=' ', index_label='x', header=['y'])

avg_reward_shield_3_action_neg_reward = sample(np.loadtxt(os.getcwd()+
    grid+'_3_neg_reward_avg_reward.data'),p=p)
plt.plot(np.convolve(avg_reward_shield_3_action_neg_reward,window(t),'same')[10:-30], label='shield-3 action-neg reward Q', c='purple')

avg_reward_shield_3_action_neg_reward = averaging(avg_reward_shield_3_action_neg_reward)
avg_reward_shield_3_action_neg_reward = pd.DataFrame(avg_reward_shield_3_action_neg_reward)
avg_reward_shield_3_action_neg_reward.to_csv(grid[1:]+'_3_neg_reward_avg_reward.dat',sep=' ', index_label='x', header=['y'])





result = pd.concat([avg_reward_no_shield, avg_reward_no_shield_neg_reward, avg_reward_shield_1_action, 
    avg_reward_shield_1_action_neg_reward, avg_reward_shield_3_action, avg_reward_shield_3_action_neg_reward], axis=1)
result.to_csv(grid[1:]+'.dat',sep=' ', index_label='x', header=['no_shield','no_shield_neg','shield_1','shield_1_neg',
  'shield_3','shield_3_neg'])
#SARSA
# avg_reward_sarsa_shield_3_action = sample(np.loadtxt(os.getcwd()+
#     grid+'_3_sarsa_avg_reward.data'),p=p)
# plt.plot(np.convolve(avg_reward_sarsa_shield_3_action[:len(avg_reward_no_shield)],window(t),'same')[10:-30], label='shield-3 action SARSA')

# avg_reward_sarsa = sample(np.loadtxt(os.getcwd()+
#     grid+'_0_sarsa_avg_reward.data'),p=p)
# plt.plot(np.convolve(avg_reward_sarsa[:len(avg_reward_no_shield)],window(t),'same')[10:-30], label='no shield SARSA')

# avg_reward_sarsa_shield_1_action = sample(np.loadtxt(os.getcwd()+
#     grid+'_1_sarsa_avg_reward.data'),p=p)
# plt.plot(np.convolve(avg_reward_sarsa_shield_1_action[:len(avg_reward_no_shield)],window(t),'same')[10:-30], label='shield-1 action SARSA')

# avg_reward_sarsa_shield_3_action_neg_reward = sample(np.loadtxt(os.getcwd()+
#     grid+'_3_neg_reward_sarsa_avg_reward.data'),p=p)
# plt.plot(np.convolve(avg_reward_sarsa_shield_3_action_neg_reward,window(t),'same')[10:-30], label='shield-3 action-neg reward SARSA')

# avg_reward_sarsa_shield_1_action_neg_reward = sample(np.loadtxt(os.getcwd()+
#     grid+'_1_neg_reward_sarsa_avg_reward.data'),p=p)
# plt.plot(np.convolve(avg_reward_sarsa_shield_1_action_neg_reward,window(t),'same')[10:-30], label='shield-1 action-neg reward SARSA')
# plt.title(title)
# plt.legend()

# plt.tight_layout()
# plt.minorticks_on()
# plt.grid(color='black', linestyle='--', linewidth=.5, which='major')
# plt.grid(color='black', linestyle='--', linewidth=.2,which='minor')
# plt.ylabel('Reward')
# plt.xlabel('Trial')
# import Image
# pp = PdfPages(title+'.pdf')
# pp.savefig()
# pp.close()
# plt.savefig("9x9.eps",bbox_inches='tight')
# plt.savefig(title+'.png')
# plt.show()
# Image.open(title+'.png').save(title+'.jpg','JPEG')


#plt.show()


# plt.plot(np.convolve(ep_len_no_shield,window(t),'same'), label='no shield')
# plt.plot(np.convolve(ep_len_shield_1_action,window(t),'same')[:45], label='shield-1 action')
# plt.plot(np.convolve(ep_len_shield_1_action_neg_reward,window(t),'same'), label='shield-1 action-neg reward')
# plt.plot(np.convolve(ep_len_shield_3_action,window(t),'same'), label='shield-3 action')
# plt.plot(np.convolve(ep_len_shield_3_action_neg_reward,window(t),'same'), label='shield-3 action-neg reward')
# plt.legend()
# plt.show()
