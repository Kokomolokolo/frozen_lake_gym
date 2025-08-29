#https://www.youtube.com/watch?v=ZhoIgo3qqLU, gym docs
import gymnasium as gym
import numpy as np

def run(episodes, is_training=True, render = False):
    env= gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human" if render else None)

    # Q-Table
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
        alpha = 0.1
        gamma = 0.95
        epsilon = 1.0
        #epsilon_decay = 0.0005
        epsilon_min = 0.05

        # Tracking
        rewards_per_episode = []
        epsilon_history = []
    else:
        data = np.load("trained_q.npz")
        q = data['q']

    
    rng = np.random.default_rng()

    for i in range(episodes):
        observation,_ = env.reset()
        episode_over = False
        total_reward = 0

        while not episode_over:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[observation,:])

            new_observation, reward, terminated, truncated, info = env.step(action)
            if is_training:
                q[observation, action] = q[observation, action] + alpha * (
                    reward + gamma * np.max(q[new_observation,:]) - q[observation, action]
                )
            episode_over = terminated or truncated
            observation = new_observation
            total_reward += reward

        if is_training:
            #epsilon = max(epsilon - epsilon_decay, 0)
            if total_reward > 0: # Das geht irgendwie aber ist schon eine traurige lösung
                if epsilon > epsilon_min:
                    epsilon *=0.999
            rewards_per_episode.append(total_reward)
            epsilon_history.append(epsilon)

            if (i + 1) % 100 == 0:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"Episode {i+1}: Avg Reward = {avg_reward:.3f}, Epsilon = {epsilon:.3f}")

    env.close()
    if is_training:
        np.savez("trained_q.npz", q=q)
        print(f"\nQ-Table gespeichert!")
        print(f"Finale Epsilon: {epsilon:.3f}")
        print(f"Durchschnittliche Belohnung (letzte 100 Episoden): {np.mean(rewards_per_episode[-100:]):.3f}")
    else:
        return q

def test_trained_agent(episodes=10, render=True):
    """Testet den trainierten Agent"""
    print("Testing trained agent...")
    run(episodes, is_training=False, render=render)

if __name__ == '__main__':
    #q = run(50000)
    #test_trained_agent()
    run(5, is_training=False, render=True)