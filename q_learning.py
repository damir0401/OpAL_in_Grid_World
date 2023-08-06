import numpy as np
from old.grid_world import GridWorld
import main
import seaborn as sbn
import matplotlib.pyplot as plt

def softmax(x):
    '''Compute softmax values of array x.

    @param x the input array
    @return the softmax array
    '''
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def update_state_action(state_action_matrix, visit_counter_matrix, position, new_position, 
                        action, reward, alpha, gamma):
    '''Return the updated utility matrix

    @param state_action_matrix the matrix before the update
    @param position the state obsrved at t
    @param new_position the state observed at t+1
    @param action the action at t
    @param new_action the action at t+1
    @param reward the reward observed after the action
    @param alpha the step size (learning rate)
    @param gamma the discount factor
    @return the updated state action matrix
    '''
    #Getting the values of Q at t and at t+1
    row = (position[0]*4) + position[1]
    q = state_action_matrix[row, action]
    row_t1 = (new_position[0]*4) + new_position[1]
    q_t1 = np.max(state_action_matrix[row_t1, :])
    #Calculate alpha based on how many times it
    #has been visited
    alpha_counted = 1.0 / (1.0 + visit_counter_matrix[row, action])
    #Applying the update rule
    #Here you can change "alpha" with "alpha_counted" if you want
    #to take into account how many times that particular state-action
    #pair has been visited until now.
    state_action_matrix[row, action] = state_action_matrix[row, action] + alpha * (reward + gamma * q_t1 - q)
    return state_action_matrix

def update_visit_counter(visit_counter_matrix, position, action):
    '''Update the visit counter
   
    Counting how many times a state-action pair has been 
    visited. This information can be used during the update.
    @param visit_counter_matrix a matrix initialised with zeros
    @param position the state observed
    @param action the action taken
    '''
    row = (position[0]*4) + position[1]
    visit_counter_matrix[row, action] += 1.0
    return visit_counter_matrix

def update_policy(policy_matrix, state_action_matrix, position):
    '''Return the updated policy matrix (q-learning)

    @param policy_matrix the matrix before the update
    @param state_action_matrix the state-action matrix
    @param position the state observed at t
    @return the updated state action matrix
    '''
    row = (position[0]*4) + position[1]
    #Getting the index of the action with the highest utility
    best_action = np.argmax(state_action_matrix[row, :])
    #Updating the policy
    policy_matrix[position[0], position[1]] = best_action
    return policy_matrix

def return_epsilon_greedy_action(policy_matrix, position, epsilon=0.1):
    tot_actions = int(np.nanmax(policy_matrix) + 1)
    tot_actions = 4
    action = int(policy_matrix[position[0], position[1]])
    non_greedy_prob = epsilon / tot_actions
    greedy_prob = 1 - epsilon + non_greedy_prob
    weight_array = np.full((tot_actions), non_greedy_prob)
    weight_array[action] = greedy_prob
    return np.random.choice(tot_actions, 1, p=weight_array)

def print_policy(policy_matrix):
    '''Print the policy using specific symbol.

    * terminal state
    ^ > v < up, right, down, left
    # obstacle
    '''
    counter = 0
    shape = policy_matrix.shape
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(policy_matrix[row,col] == -1): policy_string += " *  "            
            elif(policy_matrix[row,col] == 0): policy_string += " ^  "
            elif(policy_matrix[row,col] == 1): policy_string += " >  "
            elif(policy_matrix[row,col] == 2): policy_string += " v  "           
            elif(policy_matrix[row,col] == 3): policy_string += " <  "
            elif(np.isnan(policy_matrix[row,col])): policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)

def return_decayed_value(starting_value, global_step, decay_step):
        """Returns the decayed value.

        decayed_value = starting_value * decay_rate ^ (global_step / decay_steps)
        @param starting_value the value before decaying
        @param global_step the global step to use for decay (positive integer)
        @param decay_step the step at which the value is decayed
        """
        decayed_value = starting_value * np.power(0.1, (global_step/decay_step))
        return decayed_value



def run_q_simulation(env, rows, cols, actions, sims, total_episodes, gamma, 
                     alpha_critic):
    total_reward = []
    negative_terminal_state = []
    # exploratory_policy_matrix = np.array([[1,      1, 1, -1],
    #                                       [0, np.NaN, 0, -1],
    #                                       [0,      1, 0,  3]])
    
    # exploratory_policy_matrix = np.random.randint(low=0, high=4, size=(3, 4)).astype(np.float32)
    # exploratory_policy_matrix[1,1] = np.NaN #NaN for the obstacle at (1,1)
    # exploratory_policy_matrix[0,3] = exploratory_policy_matrix[1,3] = -1 #No action for the terminal states
    for s in range(sims):
        sim_reward = []
        policy_matrix = np.random.randint(low=0, high=4, size=(3, 4)).astype(np.float32)
        policy_matrix[1,1] = np.NaN #NaN for the obstacle at (1,1)
        policy_matrix[0,3] = policy_matrix[1,3] = -1 #No action for the terminal states
        state_action_matrix = np.zeros((rows * cols, actions)) + 0.5
        visit_counter_matrix = np.zeros((rows * cols, actions))
        sim_negative_terminal = 0
        for episode in range(total_episodes):
            #Reset and return the first observation
            position = env.reset(exploring_starts=False)
            episode_reward = 0
            is_starting = True
            for step in range(1000):
                #Take the action using epsilon-greedy
                action = return_epsilon_greedy_action(policy_matrix, position, epsilon=0.001)
                if(is_starting): 
                    action = np.random.randint(0, 4)
                    is_starting = False
                #Estimating the action through Softmax
                # row = (4*position[0]) + position[1]
                # action_array = state_action_matrix[row, :]
                # action_distribution = softmax(action_array)
                # action = np.random.choice(4, 1, p=action_distribution)
                #Move one step in the environment and get obs and reward
                new_position, reward, done, in_worst_term_state = env.step(action)
                episode_reward += reward
                if in_worst_term_state:
                    sim_negative_terminal += 1  
                #Updating the state-action matrix - Q values
                state_action_matrix = update_state_action(state_action_matrix, visit_counter_matrix, position, new_position, 
                                                        action, reward, alpha_critic, gamma)
                #Updating the policy
                policy_matrix = update_policy(policy_matrix, state_action_matrix, position)
                #Increment the visit counter
                visit_counter_matrix = update_visit_counter(visit_counter_matrix, position, action)
                position = new_position
                if done: break

            sim_reward.append(episode_reward)
        
        print_policy(policy_matrix)
        total_reward.append(sim_reward)
        negative_terminal_state.append(sim_negative_terminal)
    
    return total_reward, negative_terminal_state

def q_grid_search(env, rows, cols, actions, sims, total_episodes, gamma, 
                  alpha_range_critic):
    # Q-learning Model
    mean_matrix = [0] * len(alpha_range_critic)
    mean_neg_matrix = [0] * len(alpha_range_critic)
    for alpha_critic in alpha_range_critic:
        total_reward, negative_terminal_state = run_q_simulation(env, rows, cols, actions, sims, 
                                      total_episodes, gamma, alpha_critic)
        avg_reward = np.mean(total_reward, axis= 0)
        # print(total_reward)
        mean = np.mean(avg_reward)
        mean_matrix[alpha_range_critic.index(alpha_critic)] = mean
        
        mean_neg_term = np.mean(negative_terminal_state)
        mean_neg_matrix[alpha_range_critic.index(alpha_critic)] = mean_neg_term

    max_value = max(mean_matrix)
    max_alpha = alpha_range_critic[mean_matrix.index(max_value)]

    min_value = min(mean_neg_matrix)
    min_alpha = alpha_range_critic[mean_neg_matrix.index(min_value)]

    # Avg Reward
    f2 = plt.figure()
    plt.style.use('seaborn')
    plt.title("Q-Learning Architecture after " + str(sims) + " simulations", fontsize = 15)
    plt.xlabel("Critic's alpha range", fontsize = 15)
    plt.ylabel("Accumulated reward", fontsize = 15)
    
    plt.scatter(alpha_range_critic, mean_matrix, 
                edgecolor='black', linewidth=1, alpha=0.75)

    print("Average reward")
    print("Q-Learning: max_alpha = {}, max_value = {}".format(max_alpha, max_value))


    # Negative terminal state frequency
    print("Negative terminal state frequency")
    print("Q-Learning: min_alpha = {}, min_value = {}".format(min_alpha, min_value))

    return max_alpha, min_alpha