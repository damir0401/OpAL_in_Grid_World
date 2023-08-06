import numpy as np
from numpy import Inf
import matplotlib.pyplot as plt
from old.grid_world import GridWorld
import seaborn as sbn
import actor_critic
import opal
import q_learning
# import time

def softmax(x):
    '''Compute softmax values of array x.

    @param x the input array
    @return the softmax array
    '''
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))


def optimal_curves(env, rows, cols, actions, sims, total_episodes, gamma, 
                    max_opal_critic_alpha, max_opal_actor_alpha, max_ac_alpha, 
                    max_q_alpha):
    
    f3 = plt.figure()
    plt.style.use('seaborn')
    plt.title("Average Reward across " + str(sims) + " simulations", fontsize = 15)
    plt.xlabel("Episode", fontsize = 15)
    plt.ylabel("Accumulated reward per episode", fontsize = 15)  

    # OpAL*
    opal_total_reward, opal_neg_term_sim_list = opal.run_opal_simulation(env, rows, cols, actions, sims, 
                                                 total_episodes, gamma, 
                                                 max_opal_critic_alpha, 
                                                 max_opal_actor_alpha)

    opal_avg_reward = np.mean(opal_total_reward, axis= 0)
    episodes = np.linspace(0, total_episodes, total_episodes, endpoint=False)

    plt.scatter(episodes, opal_avg_reward, 
                edgecolor='black', linewidth=1, alpha=0.75)
    avg = sum(opal_avg_reward)/total_episodes
    print("Average for OpAL* = " + str(avg))
    plt.plot(episodes, np.full(shape=total_episodes, fill_value=avg), 
             label = "OpAL* actor/critic alpha = " + str(max_opal_critic_alpha) + \
                  ", " + str(max_opal_actor_alpha) + "\n" + "OpAL* avg reward = " + \
                    str(round(avg, 2)))

    # Original Actor Critic
    ac_total_reward, ac_neg_term_sim_list = actor_critic.run_ac_simulation(env, rows, cols, actions, sims,
                                                total_episodes, gamma, max_ac_alpha)
    
    ac_avg_reward = np.mean(ac_total_reward, axis= 0)
    plt.scatter(episodes, ac_avg_reward, 
                edgecolor='black', linewidth=1, alpha=0.75)
    avg = sum(ac_avg_reward)/total_episodes
    print("Average Reward for Actor-Critic = " + str(avg))
    plt.plot(episodes, np.full(shape=total_episodes, fill_value=avg), 
             label = "actor/critic alpha = " + str(max_ac_alpha) + "\n" + \
                "AC avg reward = " + str(round(avg, 2)))

    # Q-Learning
    q_total_reward, q_neg_term_sim_list = q_learning.run_q_simulation(env, rows, cols, actions, sims,
                                                total_episodes, gamma, max_q_alpha)
    
    q_avg_reward = np.mean(q_total_reward, axis= 0)
    plt.scatter(episodes, q_avg_reward, 
                edgecolor='black', linewidth=1, alpha=0.75)
    avg = sum(q_avg_reward)/total_episodes
    print("Average Reward for Q-Learning = " + str(avg))
    plt.plot(episodes, np.full(shape=total_episodes, fill_value=avg), 
             label = "Q-Learning alpha = " + str(max_ac_alpha) + "\n" + \
                "Q-Learning avg reward = " + str(round(avg, 2)))


    plt.legend(bbox_to_anchor=(1.04, 1), loc = "upper left", fancybox = True) 
    plt.tight_layout()


def neg_term_state_freq(env, rows, cols, actions, sims, total_episodes, gamma, 
                        min_opal_critic_alpha, min_opal_actor_alpha, min_ac_alpha,
                        min_q_alpha):
    f1 = plt.figure()
    plt.style.use('seaborn')
    plt.title("Average 0.1 Terminal State Frequency across " + str(sims) + " simulations", fontsize = 12)
    plt.xlabel("Simulation", fontsize = 15)
    plt.ylabel("Negative Terminal State Frequency", fontsize = 15)  
    
    total_sims = np.linspace(0, sims, sims, endpoint=False)

    # OpAL*
    opal_total_reward, opal_neg_term_sim_list = opal.run_opal_simulation(env, rows, cols, actions, sims,
                                                total_episodes, gamma, min_opal_critic_alpha, 
                                                 min_opal_actor_alpha)

    plt.scatter(total_sims, opal_neg_term_sim_list, edgecolor='black', linewidth=1, 
                alpha=0.75, marker="P")
    # neg_term_avg = np.mean(ac_neg_term_sim_list)
    neg_term_avg = sum(opal_neg_term_sim_list)/sims
    print("Average 0.1 Terminal State Frequency for OpAL* = " + str(neg_term_avg))
    plt.plot(total_sims, np.full(shape=sims, fill_value=neg_term_avg), 
             label = "actor/critic alpha = " + str(min_ac_alpha) + "\n" + \
                "OpAL* avg 0.1 terminal" + "\n" + "state frequency = " + str(neg_term_avg))
    
    
    # Original Actor Critic
    ac_total_reward, ac_neg_term_sim_list = actor_critic.run_ac_simulation(env, rows, cols, actions, sims,
                                                total_episodes, gamma, min_ac_alpha)
    
    plt.scatter(total_sims, ac_neg_term_sim_list, edgecolor='black', linewidth=1, 
                alpha=0.75, marker="P")
    # neg_term_avg = np.mean(ac_neg_term_sim_list)
    neg_term_avg = sum(ac_neg_term_sim_list)/sims
    print("Average 0.1 Terminal State Frequency for Actor-Critic = " + str(neg_term_avg))
    plt.plot(total_sims, np.full(shape=sims, fill_value=neg_term_avg), 
             label = "actor/critic alpha = " + str(min_ac_alpha) + "\n" + \
                "AC avg 0.1 terminal" + "\n" + "state frequency = " + str(neg_term_avg))

    # Q-Learning
    q_total_reward, q_neg_term_sim_list = q_learning.run_q_simulation(env, rows, cols, actions, sims,
                                                total_episodes, gamma, min_q_alpha)
    
    plt.scatter(total_sims, q_neg_term_sim_list, edgecolor='black', linewidth=1, 
                alpha=0.75, marker="P")
    # neg_term_avg = np.mean(ac_neg_term_sim_list)
    neg_term_avg = sum(q_neg_term_sim_list)/sims
    print("Average 0.1 Terminal State Frequency for Q-Learning = " + str(neg_term_avg))
    plt.plot(total_sims, np.full(shape=sims, fill_value=neg_term_avg), 
             label = "Q-Learning alpha = " + str(min_q_alpha) + "\n" + \
                "Q-Learning avg 0.1 terminal" + "\n" + "state frequency = " + str(neg_term_avg))

    plt.legend(bbox_to_anchor=(1.04, 1), loc = "upper left", fancybox = True) 
    plt.tight_layout()

def main():
    rows = 3
    cols = 4
    actions = 4

    env = GridWorld(rows, cols)

    #Define the state matrix
    state_matrix = np.zeros((rows, cols))
    state_matrix[0, 3] = 1
    state_matrix[1, 3] = 1
    state_matrix[1, 1] = -1
    print("State Matrix:")
    print(state_matrix)

    #Define the reward matrix
    reward_matrix = np.full((rows, cols), -0.04)
    reward_matrix[0, 3] = 1
    reward_matrix[1, 3] = 1
    print("Reward Matrix:")
    print(reward_matrix)

    #Define the reward probability matrix
    reward_prob_matrix = np.full((3, 4), 1.0)
    reward_prob_matrix[0, 3] = 0.1
    reward_prob_matrix[1, 3] = 0.3
    print("Reward Probability Matrix:")
    print(reward_prob_matrix)


    #Define the transition matrix
    # transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
    #                               [0.1, 0.8, 0.1, 0.0],
    #                               [0.0, 0.1, 0.8, 0.1],
    #                               [0.1, 0.0, 0.1, 0.8]])
    
    transition_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])

    # TO-DO: change the state-action matrix
    state_action_matrix = np.zeros((rows * cols, actions))
    print("State-Action Matrix:")
    print(state_action_matrix)

    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)
    env.setRewardProbabilityMatrix(reward_prob_matrix)

    utility_matrix = np.zeros((rows, cols))
    print("Utility Matrix:")
    print(utility_matrix)

    gamma = 0.999
    sims = 1
    total_episodes = 250

    alpha_range_critic = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_range_actor = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # OpAL*
    max_opal_critic_alpha, max_opal_actor_alpha, \
    min_opal_critic_alpha, min_opal_actor_alpha = \
                        opal.opal_grid_search(env, rows, cols, actions, sims,
                                               total_episodes, gamma, 
                                               alpha_range_critic, 
                                               alpha_range_actor)

    # # original Actor-Critic Model
    max_ac_alpha, min_ac_alpha = actor_critic.ac_grid_search(env, rows, cols, actions, sims,
                                               total_episodes, gamma, 
                                               alpha_range_critic)

    # Q-Learning Model
    max_q_alpha, min_q_alpha = q_learning.q_grid_search(env, rows, cols, actions, sims,
                                                        total_episodes, gamma, 
                                                        alpha_range_critic)

    # Optimal Curves
    # max_opal_critic_alpha, max_opal_actor_alpha, max_ac_alpha, max_q_alpha = 0.6, 1.0, 0.9, 0.2
    optimal_curves(env, rows, cols, actions, sims, total_episodes, gamma, 
                    max_opal_critic_alpha, max_opal_actor_alpha, max_ac_alpha, max_q_alpha)
    
    neg_term_state_freq(env, rows, cols, actions, sims, total_episodes, gamma, 
                        max_opal_critic_alpha, max_opal_actor_alpha, 
                        max_ac_alpha, max_q_alpha)
    
    plt.show()


if __name__ == "__main__":
    main()