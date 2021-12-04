for i in range(num_episodes):
    
    state = env.reset()
    eps*= eps_decay_factor
    done = False

    while not done:
        
        # Get the max action or random action.
        if np.random.random() < eps or np.sum(q_table[state, :]) == 0:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(q_table[state, :])

        # Execute the action and return the next state, reward and if is the las state.        
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table.
        max_q = np.max(q_table[new_state, :])
        delta_q = reward + discount_factor * max_q - q_table[state, action]
        q_table[state, action]+=  learning_rate * delta_q

        state = new_state


#
# Same thing, but with neural network.
#

states = np.identity( env.observation_space.n )

for i in range(num_episodes):


    state = env.reset()
    eps*= eps_decay_factor
    done  = False
    
    while not done:

        # Get the max action or random action.
        if np.random.random() < eps:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(
              model.predict(np.identity(env.observation_space.n)[state:state + 1]))

        # Execute the action and return the next state and reward. 
        new_state, reward, done, _ = env.step(action)

        state_input = states[new_state:new_state + 1]
        actions_pred = model.predict( state_input )
        target = reward + discount_factor * np.max( actions_pred )
        
        a = states[state:state + 1]
        b = model.predict(a)
        target_vector = b[0]
        
        target_vector[action] = target

        model.fit(
            states[state:state + 1], 
            target_vec.reshape(-1, env.action_space.n), 
            epochs = 1, 
            verbose = 0
        )

        state = new_state
