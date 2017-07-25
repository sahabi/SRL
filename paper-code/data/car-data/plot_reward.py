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

t = 10
p = 50
env = 'car'
#env = 'seaquest'

avg_reward_no_shield = sample(pd.read_csv(os.getcwd()+
    '/'+env+'-unshielded.csv'),p=p)

avg_reward_shield = sample(pd.read_csv(os.getcwd()+
    '/'+env+'-shielded.csv'),p=p)

avg_reward_shield_negreward = sample(pd.read_csv(os.getcwd()+
    '/'+env+'-shielded-negreward.csv'),p=p)

plt.plot(np.convolve(avg_reward_shield_negreward['Value'],window(t),'same')[10:-30], label='shield- neg reward')
plt.plot(np.convolve(avg_reward_no_shield['Value'],window(t),'same')[10:-30], label='no shield')
plt.plot(np.convolve(avg_reward_shield['Value'],window(t),'same')[10:-30], label='shield')

plt.legend()
plt.ylabel('Reward')
plt.xlabel('Trial')
import Image
plt.savefig('car.png')
Image.open('car.png').save('car.jpg','JPEG')
plt.show()