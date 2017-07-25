import numpy as np
import os
import pandas as pd
#pp = PdfPages('15x9_rewards.pdf')
t = 100
p = 5
#grid = '9x9_illustrative'
grid = '15x9_cycling'
grid = '/'+grid

def averaging(a):
    # a = a[:-1]
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
# avg_reward_no_shield.to_csv(grid[1:]+'_0_avg_reward.dat',sep=' ', index_label='x', header=['y'])


avg_reward_no_shield_neg_reward = sample(np.loadtxt(os.getcwd()+
    grid+'_0_huge_neg_avg_reward.data'),p=p)
# plt.plot(np.convolve(avg_reward_no_shield_neg_reward,window(t),'same')[10:-80], label='no shield Q Neg Reward', c='blue')

avg_reward_no_shield_neg_reward = averaging(avg_reward_no_shield_neg_reward)
avg_reward_no_shield_neg_reward = pd.DataFrame(avg_reward_no_shield_neg_reward)
# avg_reward_no_shield_neg_reward.to_csv(grid[1:]+'_0_huge_neg_avg_reward.dat',sep=' ', index_label='x', header=['y'])

avg_reward_shield_1_action = sample(np.loadtxt(os.getcwd()+
    grid+'_1_avg_reward.data'),p=p)
ep_len_shield_1_action = sample(np.loadtxt(os.getcwd()+
    grid+'_1_episodelen.data'),p=p)

avg_reward_shield_1_action = averaging(avg_reward_shield_1_action)
avg_reward_shield_1_action = pd.DataFrame(avg_reward_shield_1_action)
# avg_reward_shield_1_action.to_csv(grid[1:]+'_1_avg_reward.dat',sep=' ', index_label='x', header=['y'])

avg_reward_shield_1_action_neg_reward = sample(np.loadtxt(os.getcwd()+
    grid+'_1_neg_reward_avg_reward.data'),p=p)
ep_len_shield_1_action_neg_reward = sample(np.loadtxt(os.getcwd()+
    grid+'_1_neg_reward_episodelen.data'),p=p)

avg_reward_shield_1_action_neg_reward = averaging(avg_reward_shield_1_action_neg_reward)
avg_reward_shield_1_action_neg_reward = pd.DataFrame(avg_reward_shield_1_action_neg_reward)
# avg_reward_shield_1_action_neg_reward.to_csv(grid[1:]+'_1_neg_reward_avg_reward.dat',sep=' ', index_label='x', header=['y'])

avg_reward_shield_3_action = sample(np.loadtxt(os.getcwd()+
    grid+'_3_avg_reward.data'),p=p)
ep_len_shield_3_action = sample(np.loadtxt(os.getcwd()+
    grid+'_3_episodelen.data'),p=p)

avg_reward_shield_3_action = averaging(avg_reward_shield_3_action)
avg_reward_shield_3_action = pd.DataFrame(avg_reward_shield_3_action)
# avg_reward_shield_3_action.to_csv(grid[1:]+'_3_avg_reward.dat',sep=' ', index_label='x', header=['y'])

avg_reward_shield_3_action_neg_reward = sample(np.loadtxt(os.getcwd()+
    grid+'_3_neg_reward_avg_reward.data'),p=p)
ep_len_shield_3_action_neg_reward = sample(np.loadtxt(os.getcwd()+
    grid+'_3_neg_reward_episodelen.data'),p=p)

avg_reward_shield_3_action_neg_reward = averaging(avg_reward_shield_3_action_neg_reward)
avg_reward_shield_3_action_neg_reward = pd.DataFrame(avg_reward_shield_3_action_neg_reward)
# avg_reward_shield_3_action_neg_reward.to_csv(grid[1:]+'_3_neg_reward_avg_reward.dat',sep=' ', index_label='x', header=['y'])

result = pd.concat([avg_reward_no_shield, avg_reward_no_shield_neg_reward, avg_reward_shield_1_action, 
    avg_reward_shield_1_action_neg_reward, avg_reward_shield_3_action, avg_reward_shield_3_action_neg_reward], axis=1)
result.to_csv(grid[1:]+'.dat',sep=' ', index_label='x', header=['no_shield','no_shield_neg','shield_1','shield_1_neg','shield_3','shield_3_neg'])

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