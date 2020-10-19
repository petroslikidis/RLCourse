import gym
import numpy as np
env = gym.make('CliffWalking-v0')
env.reset()
gamma = 0.9
V = np.zeros(env.nS)
policy = np.zeros(env.nS)

policy_stable = False
counter = 0 

terminal_states = []
                    
while not policy_stable:
    counter += 1
    #Policy evaluation
    while True:
        delta = 0
        for s in range(env.nS):
            #print(s, V[s])
            #v = V[s]
            if s in terminal_states:
                V[s] = 0
                continue
                
            new_v = 0.0
            a = policy[s]
            for prob,s_prim,reward,done in env.P[s][a]:
                if done:
                    new_v += prob * reward
                    if not s_prim in terminal_states:
                        terminal_states.append(s_prim)
                else:
                    new_v += prob * (reward + gamma * V[s_prim])
            v = V[s]
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))
        if delta < 0.00001:
            break
            
    #Policy improvment
    policy_stable = True

    for s in range(env.nS):
        old_p = policy[s]
        
        actions = np.zeros(env.nA)
        for a in range(env.nA):
            for prob,s_prim,reward,done in env.P[s][a]:
                if done:
                    actions[a] += prob * reward 
                else:
                    actions[a] += prob * (reward + gamma * V[s_prim])
        
        policy[s] = np.argmax(actions)
        
        if policy[s] != old_p:
            policy_stable = False
        
        
print('Optimal policy')

env.reset()
done = False

state = env.s
total_reward = 0
env.render()
while not done:
    state, reward, done, info = env.step(policy[state])
    total_reward += reward
    env.render()

print('Total reward: ', total_reward)

print('Policy: ')
newP = policy
print(newP.reshape(4,12))
