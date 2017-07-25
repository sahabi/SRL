import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

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

from pylab import arange,pi,sin,cos,sqrt
import pylab
#pp = PdfPages('15x9_rewards.pdf')
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

t = 20
p = 50
#env = 'car'
env = 'seaquest'

avg_reward_no_shield = sample(pd.read_csv(os.getcwd()+
    '/'+env+'-unshielded.csv'),p=p)

avg_reward_shield = sample(pd.read_csv(os.getcwd()+
    '/'+env+'-shielded.csv'),p=p)


plt.plot(np.convolve(avg_reward_no_shield['Value'],window(t),'same')[10:-30], label='no shield',c='red')
plt.plot(np.convolve(avg_reward_shield['Value'],window(t),'same')[10:-30], label='shield',c='green')
plt.ylabel('Reward')
plt.xlabel('Trial')
plt.tight_layout()
# plt.legend()
plt.minorticks_on()
plt.grid(color='black', linestyle='--', linewidth=.5, which='major')
plt.grid(color='black', linestyle='--', linewidth=.2,which='minor')
import Image
plt.savefig('seaquest.eps',bbox_inches='tight')
Image.open('seaquest.png').save('seaquest.jpg','JPEG')
#plt.show()