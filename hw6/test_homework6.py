"""Run from the repository root: python -m unittest discover -s hw6 -v."""

import importlib.util
from pathlib import Path
import random
import sys
import unittest

import agents

SKELETON = Path(__file__).resolve().parent / 'homework6_skeleton'
spec = importlib.util.spec_from_file_location('gridworld',
                                              SKELETON / 'gridworld.py')
sys.path.insert(0, str(SKELETON))
gridworld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gridworld)
sys.path.pop(0)


class MDP:
    """A generic MDP with string states and arbitrary action names."""

    def __init__(self, transitions):
        self.transitions = transitions
        self.states = set(transitions)

    def get_actions(self, state):
        return set(self.transitions.get(state, {}))

    def get_transitions(self, state, action):
        return {end: probability for end, probability, reward
                in self.transitions[state][action]}

    def get_reward(self, state, action, end):
        return next(reward for target, probability, reward
                    in self.transitions[state][action] if target == end)


def reference_step(game, values, discount):
    """Compute a Bellman sweep independently of the agent methods."""
    result = {}
    for state in game.states:
        candidates = []
        for action in game.get_actions(state):
            total = 0.0
            outcomes = game.get_transitions(state, action)
            for end, probability in outcomes.items():
                reward = game.get_reward(state, action, end)
                total += probability * (reward
                                        + discount * values.get(end, 0.0))
            candidates.append(total)
        result[state] = max(candidates, default=0.0)
    return result


class AgentTests(unittest.TestCase):
    def test_initial_and_terminal_values(self):
        game = MDP({'s': {'exit': [('terminal', 1.0, 5.0)]}})
        for kind in (agents.ValueIterationAgent, agents.PolicyIterationAgent):
            agent = kind(game, 0.9)
            self.assertEqual(agent.get_value('s'), 0.0)
            self.assertEqual(agent.get_value('terminal'), 0.0)
            self.assertIsNone(agent.get_best_policy('terminal'))
            self.assertEqual(agent.get_q_value('s', 'exit'), 5.0)
            self.assertEqual(agent.get_best_policy('s'), 'exit')

    def test_batch_updates(self):
        game = MDP({'a': {'go': [('b', 1.0, 0.0)]},
                    'b': {'exit': [('terminal', 1.0, 10.0)]}})
        agent = agents.ValueIterationAgent(game, 0.9)
        agent.iterate()
        self.assertEqual(agent.get_value('a'), 0.0)
        self.assertEqual(agent.get_value('b'), 10.0)
        agent.iterate()
        self.assertEqual(agent.get_value('a'), 9.0)

    def test_stochastic_negative_rewards(self):
        risky = [('s', 0.25, -2.0), ('terminal', 0.75, 4.0)]
        game = MDP({'s': {'risky': risky,
                          'safe': [('terminal', 1.0, -1.0)]}})
        agent = agents.ValueIterationAgent(game, 0.8)
        self.assertEqual(agent.get_q_value('s', 'risky'), 2.5)
        for _ in range(100):
            agent.iterate()
        self.assertAlmostEqual(agent.get_value('s'), 3.125)
        self.assertEqual(agent.get_best_policy('s'), 'risky')

    def test_policy_is_fixed_during_evaluation(self):
        game = MDP({'a': {'exit': [('terminal', 1.0, 1.0)],
                          'go': [('b', 1.0, 0.0)]},
                    'b': {'exit': [('terminal', 1.0, 10.0)]}})
        agent = agents.PolicyIterationAgent(game, 0.9)
        agent.iterate()
        self.assertEqual(agent.get_value('a'), 1.0)
        self.assertEqual(agent.get_best_policy('a'), 'go')
        agent.iterate()
        self.assertEqual(agent.get_value('a'), 9.0)

    def test_policy_evaluation_converges(self):
        game = MDP({'s': {'stay': [('s', 1.0, 1.0)]}})
        agent = agents.PolicyIterationAgent(game, 0.9)
        agent.iterate()
        self.assertAlmostEqual(agent.get_value('s'), 10.0, delta=1e-5)

    def test_empty_mdp(self):
        for kind in (agents.ValueIterationAgent, agents.PolicyIterationAgent):
            agent = kind(MDP({}), 0.9)
            agent.iterate()
            self.assertEqual(agent.get_value('terminal'), 0.0)

    def test_zero_discount_and_negative_actions(self):
        game = MDP({'s': {'a': [('s', 1.0, -3.0)],
                          'b': [('s', 1.0, -1.0)]}})
        for kind in (agents.ValueIterationAgent, agents.PolicyIterationAgent):
            agent = kind(game, 0.0)
            agent.iterate()
            self.assertEqual(agent.get_value('s'), -1.0)
            self.assertEqual(agent.get_best_policy('s'), 'b')

    def test_random_generic_mdps(self):
        rng = random.Random(521)
        for case in range(20):
            transitions = {}
            states = ['a', 'b', 'c', 'd']
            for state in states:
                transitions[state] = {}
                for action in ('wait', 'jump', 'exit'):
                    probability = rng.uniform(0.1, 0.9)
                    targets = rng.sample(states + ['terminal'], 2)
                    transitions[state][action] = [
                        (targets[0], probability, rng.uniform(-5, 5)),
                        (targets[1], 1 - probability, rng.uniform(-5, 5))]
            game = MDP(transitions)
            discount = rng.uniform(0.1, 0.95)
            agent = agents.ValueIterationAgent(game, discount)
            values = dict.fromkeys(states, 0.0)
            for iteration in range(300):
                values = reference_step(game, values, discount)
                agent.iterate()
                for state in states:
                    self.assertAlmostEqual(agent.get_value(state),
                                           values[state], places=10)
            policy = agents.PolicyIterationAgent(game, discount)
            for _ in range(20):
                policy.iterate()
            for state in states:
                self.assertAlmostEqual(policy.get_value(state), values[state],
                                       delta=3e-5)

    def test_all_preset_grids(self):
        for name, layout in gridworld.PRESET_GRIDS.items():
            with self.subTest(grid=name):
                game = gridworld.Gridworld(0.2, 0.0, layout)
                value = agents.ValueIterationAgent(game, 0.9)
                policy = agents.PolicyIterationAgent(game, 0.9)
                for _ in range(500):
                    value.iterate()
                for _ in range(20):
                    policy.iterate()
                for state in game.states:
                    self.assertAlmostEqual(value.get_value(state),
                                           policy.get_value(state), delta=1e-5)

    def trace_policy(self, name, parameters):
        discount, noise, reward = parameters
        layout = gridworld.PRESET_GRIDS[name]
        game = gridworld.Gridworld(noise, reward, layout)
        agent = agents.ValueIterationAgent(game, discount)
        for _ in range(100):
            agent.iterate()
        state = next(s for s in game.states if layout[s[0]][s[1]] == 'S')
        path = []
        while state in game.states and state not in path:
            path.append(state)
            state = game._do_action(state, agent.get_best_policy(state))
        return path, state

    def test_bridge_parameters(self):
        discount, noise = agents.question_3()
        self.assertEqual(sum((discount != 0.9, noise != 0.2)), 1)
        path, end = self.trace_policy('bridge', (discount, noise, 0.0))
        self.assertEqual(end, (1, 6))
        self.assertTrue(all(row == 1 for row, col in path))

    def test_policy_parameters(self):
        for suffix, end, safe in [('a', (2, 2), False),
                                  ('b', (2, 2), True),
                                  ('c', (2, 4), False),
                                  ('d', (2, 4), True)]:
            with self.subTest(question=suffix):
                parameters = getattr(agents, 'question_4' + suffix)()
                path, actual = self.trace_policy('discount', parameters)
                self.assertEqual(actual, end)
                self.assertEqual(any(row == 0 for row, col in path), safe)
                if not safe:
                    self.assertTrue(all(row == 3 for row, col in path))
        path, end = self.trace_policy('discount', agents.question_4e())
        self.assertIn(end, path)

    def test_submission_matches_gui_agent(self):
        root = Path(__file__).resolve().parent.parent
        submission = (root / 'agents.py').read_text()
        self.assertEqual(submission, (root / 'homework6.py').read_text())
        self.assertEqual(submission, (SKELETON / 'agents.py').read_text())


if __name__ == '__main__':
    unittest.main()
