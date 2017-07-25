import numpy as np
import os
import matplotlib.pyplot as plt

t = 10
p = None

os.chdir('new')
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

avg_reward_no_shield = sample(np.loadtxt('reward-noshield.csv'),p=p)

avg_reward_shield = sample(np.loadtxt('reward-shielded.csv'),p=p)




plt.plot(np.convolve(avg_reward_no_shield,window(t)[2:-7],'same'), label='no shield')
plt.plot(np.convolve(avg_reward_shield,window(t),'same'), label='shield')

plt.legend()
import Image
plt.savefig('new_car.png')
Image.open('new_car.png').save('new_car.jpg','JPEG')
plt.show()


# plt.plot(np.convolve(ep_len_no_shield,window(t),'same'), label='no shield')
# plt.plot(np.convolve(ep_len_shield_1_action,window(t),'same')[:45], label='shield-1 action')
# plt.plot(np.convolve(ep_len_shield_1_action_neg_reward,window(t),'same'), label='shield-1 action-neg reward')
# plt.plot(np.convolve(ep_len_shield_3_action,window(t),'same'), label='shield-3 action')
# plt.plot(np.convolve(ep_len_shield_3_action_neg_reward,window(t),'same'), label='shield-3 action-neg reward')

# plt.legend()

# plt.show()