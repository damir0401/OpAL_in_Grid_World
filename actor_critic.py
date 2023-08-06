import numpy as np
from numpy import Inf
import matplotlib.pyplot as plt
from old.grid_world import GridWorld
import seaborn as sbn
import main

'''
    Original Actor-Critic functions.
'''
def update_original_critic(utility_matrix, position, new_position, 
                   reward, alpha, gamma, done):
    '''Return the updated utility matrix

    @param utility_matrix the matrix before the update
    @param observation the state obsrved at t
    @param new_observation the state observed at t+1
    @param reward the reward observed after the action
    @param alpha the step size (learning rate)
    @param gamma the discount factor
    @return the updated utility matrix
    @return the estimation error delta
    '''
    u = utility_matrix[position[0], position[1]]
    u_t1 = utility_matrix[new_position[0], new_position[1]]
    delta = reward + ((gamma * u_t1) - u)
    utility_matrix[position[0], position[1]] += alpha * delta
    return utility_matrix, delta

def update_original_actor(state_action_matrix, position, action, delta, beta_matrix=None):
    '''Return the updated state-action matrix

    @param state_action_matrix the matrix before the update
    @param observation the state obsrved at t
    @param action taken at time t
    @param delta the estimation error returned by the critic
    @param beta_matrix a visit counter for each state-action pair
    @return the updated matrix
    '''
    row = (position[0]*4) + position[1]
    if beta_matrix is None: beta = 1
    else: beta = 1 / beta_matrix[row, action]
    state_action_matrix[row, action] += beta * delta
    return state_action_matrix

def run_ac_simulation(env, rows, cols, actions, sims, total_episodes, gamma, alpha_critic):
    total_reward = []
    negative_terminal_state = []
    for s in range(sims):
        sim_reward = []
        state_action_matrix = np.zeros((rows * cols, actions)) + 0.5
        utility_matrix = np.zeros((rows, cols)) + 0.5
        beta_matrix = np.zeros((rows * cols, actions))
        sim_negative_terminal = 0
        for episode in range(total_episodes):
            #Reset and return the first observation
            position = env.reset(exploring_starts=False)
            episode_reward = 0
            for step in range(1000):
                #Estimating the action through Softmax
                row = (position[0] * 4) + position[1]
                action_array = state_action_matrix[row, :]
                action_distribution = main.softmax(action_array)
                action = np.random.choice(4, 1, p=action_distribution)
                #To enable the beta parameter, enable the line below
                #and add beta_matrix=beta_matrix in the update actor function
                beta_matrix[row, action] += 1 #increment the counter
                #Move one step in the environment and get obs and reward
                new_position, reward, done, in_worst_term_state = env.step(action)
                episode_reward += reward
                if in_worst_term_state:
                    sim_negative_terminal += 1
                utility_matrix, delta = update_original_critic(utility_matrix, position, 
                                                    new_position, reward, alpha_critic, gamma, done)
                state_action_matrix = update_original_actor(state_action_matrix, position, 
                                                action, delta, beta_matrix=None)
                position = new_position
                if done: break

            sim_reward.append(episode_reward)

        total_reward.append(sim_reward)
        negative_terminal_state.append(sim_negative_terminal)
        # print(state_action_matrix)
    
    return total_reward, negative_terminal_state

def ac_grid_search(env, rows, cols, actions, sims, total_episodes, gamma, alpha_range_critic):
    # original Actor-Critic Model
    mean_matrix = [0] * len(alpha_range_critic)
    mean_neg_matrix = [0] * len(alpha_range_critic)
    for alpha_critic in alpha_range_critic:
        total_reward, negative_terminal_state = run_ac_simulation(env, rows, cols, actions, sims, 
                                      total_episodes, gamma, alpha_critic)
        avg_reward = np.mean(total_reward, axis= 0)
        mean = np.mean(avg_reward)
        mean_matrix[alpha_range_critic.index(alpha_critic)] = mean
        
        mean_neg_term = np.mean(negative_terminal_state)
        mean_neg_matrix[alpha_range_critic.index(alpha_critic)] = mean_neg_term

    max_value = max(mean_matrix)
    max_alpha = alpha_range_critic[mean_matrix.index(max_value)]

    min_value = min(mean_neg_matrix)
    print(mean_neg_matrix)
    min_alpha = alpha_range_critic[mean_neg_matrix.index(min_value)]

    # Avg Reward
    f2 = plt.figure()
    plt.style.use('seaborn')
    plt.title("Original Actor-Critic Architecture after " + str(sims) + " simulations", fontsize = 15)
    plt.xlabel("Critic's alpha range", fontsize = 15)
    plt.ylabel("Accumulated reward", fontsize = 15)
    
    plt.scatter(alpha_range_critic, mean_matrix, 
                edgecolor='black', linewidth=1, alpha=0.75)

    print("Average reward")
    print("Actor-critic: max_alpha = {}, max_value = {}".format(max_alpha, max_value))

    # Negative terminal state frequency
    print("Negative terminal state frequency")
    print("Actor-critic: min_alpha = {}, min_value = {}".format(min_alpha, min_value))

    return max_alpha, min_alpha
