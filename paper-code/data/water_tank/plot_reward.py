import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pylab
from pylab import arange,pi,sin,cos,sqrt
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
p = 100
grid = 'watertank'
grid = '/'+grid

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

#Q_0
avg_reward_q_no_shield = sample(np.loadtxt(os.getcwd()+
    grid+'_no_shield_avg_reward.data'),p=p)
plt.plot(np.convolve(avg_reward_q_no_shield,window(t),'same')[:-80], label='Q',c='red')

avg_reward_q_shield = sample(np.loadtxt(os.getcwd()+
    grid+'_shield_avg_reward.data'),p=p)
plt.plot(np.convolve(avg_reward_q_shield,window(t),'same')[:-80], label='Q shielded',c='green')

#SARSA
avg_reward_sarsa_no_shield = sample(np.loadtxt(os.getcwd()+
    grid+'_sarsa_no_shield_avg_reward.data'),p=p)
plt.plot(np.convolve(avg_reward_sarsa_no_shield,window(t),'same')[:-80], linestyle=':',  label='SARSA',c='cyan')

avg_reward_q_shield = sample(np.loadtxt(os.getcwd()+
    grid+'_sarsa_shield_avg_reward.data'),p=p)
plt.plot(np.convolve(avg_reward_q_shield,window(t),'same')[:-80], linestyle=':', label='SARSA shielded',c='magenta')


plt.tight_layout()
# plt.legend()
plt.minorticks_on()
plt.grid(color='black', linestyle='--', linewidth=.5, which='major')
plt.grid(color='black', linestyle='--', linewidth=.2,which='minor')
plt.ylabel('Reward')
plt.xlabel('Trial')
# plt.title('Watertank', fontsize=20)
import Image
plt.savefig('watertank.eps',bbox_inches='tight')
plt.show()
Image.open('watertank.png').save('watertanks.jpg','JPEG')

# plt.plot(np.convolve(ep_len_no_shield,window(t),'same'), label='no shield')
# plt.plot(np.convolve(ep_len_shield_1_action,window(t),'same')[:45], label='shield-1 action')
# plt.plot(np.convolve(ep_len_shield_1_action_neg_reward,window(t),'same'), label='shield-1 action-neg reward')
# plt.plot(np.convolve(ep_len_shield_3_action,window(t),'same'), label='shield-3 action')
# plt.plot(np.convolve(ep_len_shield_3_action_neg_reward,window(t),'same'), label='shield-3 action-neg reward')
# plt.legend()
# plt.show()