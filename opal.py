import numpy as np
from numpy import Inf
import matplotlib.pyplot as plt
from old.grid_world import GridWorld
import seaborn as sbn
import main

'''
    OpAL* functions.
'''
def update_critic(utility_matrix, position, new_position, 
                   reward, alpha_critic, gamma, done):
    '''Return the updated utility matrix

    @param utility_matrix the matrix before the update
    @param position the state observed at t
    @param new_position the state observed at t+1
    @param reward the reward observed after the action
    @param alpha the step size (learning rate)
    @param gamma the discount factor
    @return the updated utility matrix
    @return the estimation error delta
    '''
    u = utility_matrix[position[0], position[1]]
    u_t1 = utility_matrix[new_position[0], new_position[1]]
    delta = reward + ((gamma * u_t1) - u)
    utility_matrix[position[0], position[1]] += alpha_critic * delta
    # print(delta)
    return utility_matrix, delta

def update_actor(state_action_matrix, position, action, delta, g_matrix, n_matrix,
                 alpha_actor, beta_matrix=None):
    '''Return the updated state-action matrix

    @param state_action_matrix the matrix before the update
    @param position the state observed at t
    @param action taken at time t
    @param delta the estimation error returned by the critic
    @param beta_matrix a visit counter for each state-action pair
    @return the updated matrix
    '''
    row = (position[0]*4) + position[1]
    if beta_matrix is None: beta = 1 
    else: beta = 1 / beta_matrix[row, action]
    
    # anneal the learning rate
    g_alpha = beta * alpha_actor
    n_alpha = beta * alpha_actor

    # normalize PE
    range = 2
    f_delta = delta/range

    g_matrix[row, action] += g_alpha * g_matrix[row, action] * f_delta
    n_matrix[row, action] += n_alpha * n_matrix[row, action] * (-1) * f_delta

    rho = 0 # rho = -1, -.5, 0, .5, 1
    # rho = -1
    state_action_matrix[row, action] = (1 + rho) * g_matrix[row, action] - (1 - rho) * n_matrix[row, action]

    return state_action_matrix

def run_opal_simulation(env, rows, cols, actions, sims, total_episodes, gamma, 
                        alpha_critic, alpha_actor):
    total_reward = []
    negative_terminal_state = []
    for s in range(sims):
        sim_reward = []
        state_action_matrix = np.zeros((rows * cols, actions)) + 0.5
        utility_matrix = np.zeros((rows, cols)) + .5
        g_matrix = np.ones((rows * cols, actions))
        n_matrix = np.ones((rows * cols, actions))
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
                utility_matrix, delta = update_critic(utility_matrix, position, 
                                                    new_position, reward, alpha_critic, gamma, done)
                state_action_matrix = update_actor(state_action_matrix, position, 
                                                action, delta, g_matrix, n_matrix, alpha_actor, beta_matrix=None)
                position = new_position
                if done: break

            sim_reward.append(episode_reward)

        total_reward.append(sim_reward)
        negative_terminal_state.append(sim_negative_terminal)
    
    return total_reward, negative_terminal_state

def opal_grid_search(env, rows, cols, actions, sims, total_episodes, gamma, 
                     alpha_range_critic, alpha_range_actor):
    mean_matrix = np.matrix(np.zeros((len(alpha_range_critic), len(alpha_range_actor))))
    mean_neg_matrix = np.matrix(np.zeros((len(alpha_range_critic), len(alpha_range_actor))))
    for alpha_critic in alpha_range_critic:
        for alpha_actor in alpha_range_actor:
            total_reward, negative_terminal_state = run_opal_simulation(env, rows, cols, actions, sims, 
                                               total_episodes, gamma, 
                                               alpha_critic, alpha_actor)
            avg_reward = np.mean(total_reward, axis= 0)    
            mean = np.mean(avg_reward)
            mean_matrix[alpha_range_critic.index(alpha_critic), 
                        alpha_range_actor.index(alpha_actor)] = mean
            
            mean_neg_term = np.mean(negative_terminal_state)
            mean_neg_matrix[alpha_range_critic.index(alpha_critic),
                            alpha_range_actor.index(alpha_actor) ] = mean_neg_term
    
    max_value = mean_matrix.max()
    max_opal_alpha = (0, 0)

    for i in range(len(alpha_range_critic)):
        for j in range(len(alpha_range_actor)):
            if mean_matrix[i, j] == max_value:
                max_opal_alpha = (alpha_range_critic[i], alpha_range_actor[j])

    min_value = mean_neg_matrix.min()
    min_opal_alpha = (0, 0)

    for i in range(len(alpha_range_critic)):
        for j in range(len(alpha_range_actor)):
            if mean_neg_matrix[i, j] == min_value:
                min_opal_alpha = (alpha_range_critic[i], alpha_range_actor[j])


    # Avg reward
    f1 = plt.figure()
    heatmap = sbn.heatmap(mean_matrix, annot=True, linewidth=0.5, 
                        xticklabels=alpha_range_actor, 
                        yticklabels=alpha_range_critic)
    heatmap.set(title="Grid Search over actor/critic alpha", 
                xlabel="Actor alpha values", ylabel="Critic alpha values") 
    
    print("Average reward")
    print("OpAL*: max_opal_critic_alpha = {}, max_opal_actor_alpha = {}, max_value = {}".format(
                                        max_opal_alpha[0], max_opal_alpha[1], max_value))
    
    # Negative terminal state frequency
    print("Negative terminal state frequency")
    print("OpAL*: min_alpha = {}, min_value = {}".format(min_opal_alpha, min_value))
    print("")

    return max_opal_alpha[0], max_opal_alpha[1], min_opal_alpha[0], min_opal_alpha[1]
