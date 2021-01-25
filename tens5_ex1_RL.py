import gym
import numpy as np
import time

##########################################################################################
##########################################################################################
# GYM INTERACTION

env = gym.make('FrozenLake-v0')  # we are going to use the FrozenLake enviornment


print(env.observation_space.n)   # get number of states
print(env.action_space.n)   # get number of actions

env.reset()  # reset enviornment to default state

for i in range(10):

	action = env.action_space.sample()  # get a random action 

	env.render()   # render the GUI for the enviornment 

	(new_state, reward, done, info) = env.step(action)  # take action, notice it returns information about the action


##########################################################################################
##########################################################################################
# ALGORITHM


STATES = env.observation_space.n   # get number of states
ACTIONS = env.action_space.n   # get number of actions

Q = np.zeros((STATES, ACTIONS))  # create a matrix with all 0 values 


# CONSTANTS
EPISODES = 2000 # how many times to run the enviornment from the beginning
MAX_STEPS = 100  # max number of steps allowed for each run of enviornment

LEARNING_RATE = 0.81  # learning rate alfa
GAMMA = 0.96

epsilon = 0.9  # exploration vs exploitation -> 90% random

RENDER = False # if you want to see training set to true


rewards = []
for episode in range(EPISODES):

	state = env.reset()
	
	for _ in range(MAX_STEPS):
    
		if RENDER:
			env.render()

		# exploration vs exploitation
		if np.random.uniform(0, 1) < epsilon:
			action = env.action_space.sample()  
		else:
			action = np.argmax(Q[state, :])  # bira akciju iz Q tablice

		(next_state, reward, done, _) = env.step(action)

		Q[state, action] += LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])  # max bira najveći Q, gamma bira jel bitnije trenutačno ili buduće
																										# ako je trenutačni Q velik, a budući nije, želimo smanjit trenutačni
		state = next_state																					# pa oduzmemo da ode u negativu

		if done: 
			rewards.append(reward)
			epsilon -= 0.001     # pomalo smanjujemo exploraciju
			break  # reached goal

print(Q)
print(f"Average reward: {sum(rewards)/len(rewards)}:")
# and now we can see our Q values!

##########################################################################################
##########################################################################################
# PLOTTED PROGRESS

# we can plot the training progress and see how the agent improved
import matplotlib.pyplot as plt

def get_average(values):
	return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100):
	avg_rewards.append(get_average(rewards[i:i+100])) 

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()

