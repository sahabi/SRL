import numpy as np
import os
import pandas as pd
#pp = PdfPages('15x9_rewards.pdf')
t = 100
p = 100
#grid = '9x9_illustrative'
grid = 'seaquest'
grid = '/'+grid

def averaging(a):
    a = a[:]
    n_samples = len(a)/100.
    print n_samples
    if len(a)%100. == 0:
        a = a.reshape(100,int(n_samples))
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

avg_reward_no_shield = pd.DataFrame.from_csv(os.getcwd()+
    grid+'-shielded.csv')

# print avg_reward_no_shield['Value'].values

avg_reward_no_shield = averaging(avg_reward_no_shield['Value'].values)
avg_reward_no_shield = pd.DataFrame(avg_reward_no_shield)
# avg_reward_no_shield.to_csv(grid[1:]+'_0_avg_reward.dat',sep=' ', index_label='x', header=['y'])

avg_reward_shield_1_action = pd.DataFrame.from_csv(os.getcwd()+
    grid+'-unshielded.csv')

avg_reward_shield_1_action = averaging(avg_reward_shield_1_action['Value'].values)
avg_reward_shield_1_action = pd.DataFrame(avg_reward_shield_1_action)
# avg_reward_shield_3_action.to_csv(grid[1:]+'_3_avg_reward.dat',sep=' ', index_label='x', header=['y'])

result = pd.concat([avg_reward_no_shield, avg_reward_shield_1_action], axis=1)
result.to_csv(grid[1:]+'.dat',sep=' ', index_label='x', header=['no_shield','shield_1'])

# plt.plot(np.convolve(avg_reward_no_shield,window(t),'same')[10:-30], label='no shield Q', c='red')
# plt.plot(np.convolve(avg_reward_shield_1_action,window(t),'same')[10:-30], label='shield-1 action',c='orange')
# plt.plot(np.convolve(avg_reward_shield_1_action_neg_reward,window(t),'same')[10:-30], label='shield-1 action-neg reward',c='green')
# plt.plot(np.convolve(avg_reward_shield_3_action,window(t),'same')[10:-30], label='shield-3 action',c='gray')
# plt.plot(np.convolve(avg_reward_shield_3_action_neg_reward,window(t),'same')[10:-30], label='shield-3 action-neg reward',c='purple')

# plt.tight_layout()
# # plt.legend()
# plt.minorticks_on()
# plt.grid(color='black', linestyle='--', linewidth=.5, which='major')
# plt.grid(color='black', linestyle='--', linewidth=.2,which='minor')
# plt.ylabel('Reward')
# plt.xlabel('Trial')
# import Image
# plt.savefig('15x9_rewards.eps',bbox_inches='tight')
# Image.open('15x9_rewards.png').save('15x9_rewards.jpg','JPEG')
# plt.show()


# plt.plot(np.convolve(ep_len_no_shield,window(t),'same'), label='no shield')
# plt.plot(np.convolve(ep_len_shield_1_action,window(t),'same')[:45], label='shield-1 action')
# plt.plot(np.convolve(ep_len_shield_1_action_neg_reward,window(t),'same'), label='shield-1 action-neg reward')
# plt.plot(np.convolve(ep_len_shield_3_action,window(t),'same'), label='shield-3 action')
# plt.plot(np.convolve(ep_len_shield_3_action_neg_reward,window(t),'same'), label='shield-3 action-neg reward')

# plt.legend()

# plt.show()