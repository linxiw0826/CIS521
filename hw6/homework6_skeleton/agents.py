# Include your imports here, if any are used.

student_name = "Linxi Wu"


# 1. Value Iteration
class ValueIterationAgent:
    """Implement Value Iteration Agent using Bellman Equations."""

    def __init__(self, game, discount):
        """Store game object and discount value into the agent object,
        initialize values if needed.
        """
        self.game = game
        self.discount = discount
        self.values = {state: 0.0 for state in game.states}

    def get_value(self, state):
        """Return value V*(s) correspond to state.
        State values should be stored directly for quick retrieval.
        """
        return self.values.get(state, 0.0)

    def get_q_value(self, state, action):
        """Return Q*(s,a) correspond to state and action.
        Q-state values should be computed using Bellman equation:
        Q*(s,a) = Σ_s' T(s,a,s') [R(s,a,s') + γ V*(s')]
        """
        return sum(
            probability * (self.game.get_reward(state, action, next_state)
                           + self.discount * self.get_value(next_state))
            for next_state, probability in
            self.game.get_transitions(state, action).items()
        )

    def get_best_policy(self, state):
        """Return policy π*(s) correspond to state.
        Policy should be extracted from Q-state values using policy extraction:
        π*(s) = argmax_a Q*(s,a)
        """
        return max(self.game.get_actions(state),
                   key=lambda action: self.get_q_value(state, action),
                   default=None)

    def iterate(self):
        """Run single value iteration using Bellman equation:
        V_{k+1}(s) = max_a Q*(s,a)
        Then update values: V*(s) = V_{k+1}(s)
        """
        # Read every successor value from the previous iteration.
        self.values = {
            state: max((self.get_q_value(state, action)
                        for action in self.game.get_actions(state)),
                       default=0.0)
            for state in self.game.states
        }


# 2. Policy Iteration
class PolicyIterationAgent(ValueIterationAgent):
    """Implement Policy Iteration Agent.

    The only difference between policy iteration and value iteration is at
    their iteration method. However, if you need to implement helper function
    or override ValueIterationAgent's methods, you can add them as well.
    """

    def iterate(self):
        """Run single policy iteration.
        Fix current policy, iterate state values V(s) until
        |V_{k+1}(s) - V_k(s)| < ε
        """
        epsilon = 1e-6

        policy = {state: self.get_best_policy(state)
                  for state in self.game.states}
        while True:
            values = {
                state: (self.get_q_value(state, action)
                        if action is not None else 0.0)
                for state, action in policy.items()
            }
            difference = max((abs(values[state] - self.get_value(state))
                              for state in values), default=0.0)
            self.values = values
            if difference < epsilon:
                break


# 3. Bridge Crossing Analysis
def question_3():
    discount = 0.9
    noise = 0.0
    return discount, noise


# 4. Policies
def question_4a():
    discount = 0.2
    noise = 0.0
    living_reward = -1.0
    return discount, noise, living_reward


def question_4b():
    discount = 0.2
    noise = 0.2
    living_reward = -0.1
    return discount, noise, living_reward


def question_4c():
    discount = 0.9
    noise = 0.0
    living_reward = -0.1
    return discount, noise, living_reward


def question_4d():
    discount = 0.9
    noise = 0.2
    living_reward = -0.1
    return discount, noise, living_reward


def question_4e():
    discount = 0.9
    noise = 0.0
    living_reward = 2.0
    return discount, noise, living_reward


# 5. Feedback
# Just an approximation is fine.
feedback_question_1 = """
1
"""

feedback_question_2 = """
The most challenging part was keeping the policy fixed during policy
evaluation and using the previous values for every Bellman update.
"""

feedback_question_3 = """
I liked seeing how discount, noise, and living reward change the optimal
policy. More small examples of policy iteration would be helpful.
"""
