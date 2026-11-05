import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """
    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    # --- STEP 0: Initialize forward message 
    forward_messages[0] = rover.Distribution()
    x0_y0 = observations[0]
    for z_k in all_possible_hidden_states:
        p_x0_y0 = observation_model(z_k)[x0_y0] if x0_y0 is not None else 1.0
        forward_messages[0][z_k] = p_x0_y0 * prior_distribution[z_k]
    forward_messages[0].renormalize()
    
    # step1
    for i in range(1, num_time_steps):
        forward_messages[i] = rover.Distribution() 
        xi_yi = observations[i]
        for z_prev in all_possible_hidden_states:
            alpha_prev = forward_messages[i-1][z_prev]
            if alpha_prev == 0:
                continue # Skip states with 0 probability
        
            for z_curr, p_trans in transition_model(z_prev).items():
                if p_trans > 0:
                    if z_curr not in forward_messages[i]:
                        forward_messages[i][z_curr] = 0.0
                    forward_messages[i][z_curr] += alpha_prev * p_trans

        # apply the observation model
        for z_curr in all_possible_hidden_states:
            current_val = forward_messages[i].get(z_curr, 0.0)
            p_obs = observation_model(z_curr)[xi_yi] if xi_yi is not None else 1.0
            forward_messages[i][z_curr] = current_val * p_obs
            
        forward_messages[i].renormalize()

    # step2
    backward_messages[num_time_steps-1] = rover.Distribution()
    for z_k in all_possible_hidden_states:
        backward_messages[num_time_steps-1][z_k] = 1.0

    for i in reversed(range(num_time_steps - 1)):        
        backward_messages[i] = rover.Distribution()
        xi_yi_plus1 = observations[i+1]
        
        for z_curr in all_possible_hidden_states:
            total = 0.0
            # loop through valid next states
            for z_next, p_trans in transition_model(z_curr).items():
                if p_trans > 0:
                    p_obs = observation_model(z_next)[xi_yi_plus1] if xi_yi_plus1 is not None else 1.0
                    total += backward_messages[i+1][z_next] * p_obs * p_trans
            backward_messages[i][z_curr] = total
            
        backward_messages[i].renormalize()

    # -step3
    for i in range(num_time_steps):
        marginals[i] = rover.Distribution()
        for z_k in all_possible_hidden_states:
            marginals[i][z_k] = forward_messages[i][z_k] * backward_messages[i][z_k]
        marginals[i].renormalize() 

    print("\nSanity check at time step 30:")
    for state, prob in marginals[30].items():
        if prob > 0:
            print(state, prob)    
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """
    num_time_steps = len(observations)
    weights = [None] * num_time_steps
    psi = [{} for _ in range(num_time_steps)] 
    
    weights[0] = rover.Distribution()
    x0_y0 = observations[0]
    for z_k in all_possible_hidden_states:
        p_obs = observation_model(z_k)[x0_y0] if x0_y0 is not None else 1.0
        log_p_obs = np.log(p_obs) if p_obs > 0 else -float('inf')
        log_p_prior = np.log(prior_distribution[z_k]) if prior_distribution[z_k] > 0 else -float('inf')
        weights[0][z_k] = log_p_obs + log_p_prior

    for n_plus1 in range(1, num_time_steps):
        weights[n_plus1] = rover.Distribution()
        xy_nplus1 = observations[n_plus1]
        
        # Track the max transitions
        z_n_max_trackers = {zk: -float('inf') for zk in all_possible_hidden_states}
        z_n_max_states = {zk: None for zk in all_possible_hidden_states}
        
        for z_n in all_possible_hidden_states:
            wn_zn = weights[n_plus1-1][z_n]
            if wn_zn == -float('inf'):
                continue
                
            for zk_nplus1, p_trans in transition_model(z_n).items():
                if p_trans > 0:
                    val = wn_zn + np.log(p_trans)
                    if val > z_n_max_trackers[zk_nplus1]:
                        z_n_max_trackers[zk_nplus1] = val
                        z_n_max_states[zk_nplus1] = z_n
                        
    
        for zk_nplus1 in all_possible_hidden_states:
            p_obs = observation_model(zk_nplus1)[xy_nplus1] if xy_nplus1 is not None else 1.0
            log_p_obs = np.log(p_obs) if p_obs > 0 else -float('inf')
            
            psi[n_plus1][zk_nplus1] = z_n_max_states[zk_nplus1]
            weights[n_plus1][zk_nplus1] = log_p_obs + z_n_max_trackers[zk_nplus1]

    # backtrack
    z_hat_N = max(all_possible_hidden_states, key=lambda z: weights[num_time_steps-1][z])
    
    estimated_hidden_states = [None] * num_time_steps
    estimated_hidden_states[-1] = z_hat_N
    
    for i in reversed(range(num_time_steps - 1)):
        estimated_hidden_states[i] = psi[i+1][estimated_hidden_states[i+1]]
        
    return estimated_hidden_states

def get_error(estimated_hidden_states, marginals):
    P_squiggly = 0
    P_arrow = 0
    total_squiggly = 0
    total_arrow = 0

    marginal_states = [None] * num_time_steps
    for i in range(num_time_steps):
        max_p_i_zk = -float('inf')
        max_zk = None
        
        for z_k in all_possible_hidden_states:
            p_i_zk = marginals[i][z_k]
            if p_i_zk > max_p_i_zk:
                max_p_i_zk = p_i_zk
                max_zk = z_k
        
        marginal_states[i] = max_zk
    
    for i in range(num_time_steps):
        if estimated_hidden_states[i] == hidden_states[i]:
            total_squiggly += 1
        if marginal_states[i] == hidden_states[i]:
            total_arrow += 1
            
    P_squiggly = 1 - (total_squiggly / 100)
    P_arrow = 1 - (total_arrow / 100)

    return P_squiggly, P_arrow, marginal_states

def print_z_arrow(marginal_state_sequence, transition_model):
    for i in range(num_time_steps - 1): # Fixed index out of bounds error
        z_i = marginal_state_sequence[i]
        z_iplus1 = marginal_state_sequence[i+1]
        if transition_model(z_i)[z_iplus1] == 0:
            print("invalid, zi, ziplus1", z_i, z_iplus1, "i:", i)

if __name__ == '__main__':
   
    enable_graphics = True
    missing_observations = True
    # missing_observations = False #changed for part1

    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')

    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
        
    error1, error2, marginal_state_sequence = get_error(estimated_states, marginals)
    print(error1, error2)
    print_z_arrow(marginal_state_sequence, rover.transition_model)
        
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()