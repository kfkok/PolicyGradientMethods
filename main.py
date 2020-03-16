import numpy as np
from methods import OneStepActorCritic
import gym
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
EPISODES = 1500
agent = OneStepActorCritic(state_size, action_size)
rand = np.random
actions_set = np.arange(action_size)
total_rewards_overtime = []

for e in range(EPISODES):
    done = False
    total_rewards = 0
    state = env.reset()

    while not done:
        if (e % 20 == 0) and e > 0:
            env.render()

        action_probabilities = agent.get_action_probabilities(state).flatten()
        action = rand.choice(actions_set, p=action_probabilities)
        next_state, reward, done, _ = env.step(action)
        agent.train(state, action, next_state, reward, done)

        total_rewards += reward
        state = next_state

        if done:
            print("ep:", e, "total rewards:", total_rewards)
            total_rewards_overtime.append(total_rewards)

# plot reward graph over time
total_rewards_overtime = np.array(total_rewards_overtime)
y = np.mean(total_rewards_overtime.reshape(-1, 10), axis=1)
x = (np.array(np.arange(len(y))) * 10) + 5
fig, ax = plt.subplots()
plt.title('Average reward over time', )
ax.set_aspect('equal')
ax.set_xlabel('episodes')
ax.set_ylabel('reward')
ax.set_ylim((0, 200))
ax.set_xlim((0, EPISODES))
ax.plot(x, y, color='red', linewidth=0.5, marker='o', markersize=3)
plt.show()