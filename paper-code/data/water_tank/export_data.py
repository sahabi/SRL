import numpy as np
import os
import pandas as pd
#pp = PdfPages('15x9_rewards.pdf')
t = 1
p = 100
#grid = '9x9_illustrative'
grid = 'watertank'
grid = '/'+grid

def averaging(a,out_samples=1000.):
    a = a[:-1]
    n_samples = len(a)/out_samples
    print n_samples
    a = a.reshape(int(out_samples),int(n_samples))
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
    grid+'_no_shield_avg_reward.data'),p=p)

avg_reward_no_shield = averaging(avg_reward_no_shield)
avg_reward_no_shield = pd.DataFrame(avg_reward_no_shield)
# avg_reward_no_shield.to_csv(grid[1:]+'_0_avg_reward.dat',sep=' ', index_label='x', header=['y'])


avg_reward_shield = sample(np.loadtxt(os.getcwd()+
    grid+'_shield_avg_reward.data'),p=p)
# plt.plot(np.convolve(avg_reward_no_shield_neg_reward,window(t),'same')[10:-80], label='no shield Q Neg Reward', c='blue')

avg_reward_shield = averaging(avg_reward_shield)
avg_reward_shield = pd.DataFrame(avg_reward_shield)
# avg_reward_no_shield_neg_reward.to_csv(grid[1:]+'_0_huge_neg_avg_reward.dat',sep=' ', index_label='x', header=['y'])

avg_reward_sarsa_shield = sample(np.loadtxt(os.getcwd()+
    grid+'_sarsa_shield_avg_reward.data'),p=p)

avg_reward_sarsa_shield = averaging(avg_reward_sarsa_shield)
avg_reward_sarsa_shield = pd.DataFrame(avg_reward_sarsa_shield)

avg_reward_sarsa_no_shield = sample(np.loadtxt(os.getcwd()+
    grid+'_sarsa_no_shield_avg_reward.data'),p=p)

avg_reward_sarsa_no_shield = averaging(avg_reward_sarsa_no_shield)
avg_reward_sarsa_no_shield = pd.DataFrame(avg_reward_sarsa_no_shield)

result = pd.concat([avg_reward_no_shield,avg_reward_shield, avg_reward_sarsa_no_shield, avg_reward_sarsa_shield], axis=1)
result.to_csv(grid[1:]+'.dat',sep=' ', index_label='x', header=['no_shield','shield_1','no_shield_sarsa','shield_1_sarsa'])

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