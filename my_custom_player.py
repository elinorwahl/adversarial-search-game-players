
import math
import random
import time
from sample_players import DataPlayer

   
"""
    **********************************************************************
                          Minimax Alpha-Beta Player
    **********************************************************************
"""

class MinimaxAlphaBetaPlayer(DataPlayer):
    """ Implement a player that combines minimax with alpha-beta pruning
    to play knight's Isolation.
    """
    def get_action(self, state):
        """ Randomly select a move as player 1 or 2 on an empty board, otherwise
        return the optimal minimax move at a fixed search depth of 5 plies.
        """
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            depth_limit = 5
            for depth in range(1, depth_limit + 1):
                self.queue.put(self.minimax_alpha_beta(state, depth))              

    def minimax_alpha_beta(self, state, depth):
        """ Return the move along a branch of the game tree that has the
        best possible value.
        """

        def min_value(state, alpha, beta, depth):
            """ Return the value for a win (+1) if the game is over,
            otherwise return the minimum value over all legal child nodes.
            """
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), alpha, beta, depth - 1))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value

        def max_value(state, alpha, beta, depth):
            """ Return the value for a loss (-1) if the game is over,
            otherwise return the maximum value over all legal child nodes.
            """
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), alpha, beta, depth - 1))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), float("-inf"), 
                                                            float("inf"), depth - 1))

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

"""
    **********************************************************************
              Monte Carlo Upper Confidence Tree Search Player
    **********************************************************************
"""

class Node():
    """ Implement a class that saves the states of game move values,
    child nodes, parent nodes, and expansions along the search tree.

    Adapted from the Connect 4 player here:
    https://github.com/Alfo5123/Connect4/blob/master/game.py
    """
    def __init__(self, state, parent=None):
        self.visits = 1
        self.value = 0.0
        self.state = state
        self.children = []
        self.child_actions = []
        self.parent = parent

    def add_child(self, child_state, action):
        child = Node(child_state, self)
        self.children.append(child)
        self.child_actions.append(action)

    def update(self, value):
        self.value += value
        self.visits += 1

    def fully_explored(self):
        if len(self.child_actions) == len(self.state.actions()):
            return True
        return False
        

class MonteCarloUCTPlayer(DataPlayer):
    """ Implement a player that invokes the Node class and combines
    Monte Carlo tree search with an upper confidence tree heuristic
    to play knight's Isolation.

    Adapted from the Connect 4 player here:
    https://github.com/Alfo5123/Connect4/blob/master/game.py
    """
    def get_action(self, state):
        """ Randomly select a move as player 1 or 2 on an empty board, 
        otherwise return the optimal Monte Carlo move.
        """
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.monte_carlo_uct(state))
    
    def expand(self, node):
        """ Select a legal move and advance to the corresponding child node.
        """
        possible_actions = node.state.actions()
        child_action_attempts = [action for action in node.child_actions]
        for action in possible_actions:
            if action not in child_action_attempts:
                new_state = node.state.result(action)
                break
        node.add_child(new_state, action)
        return node.children[-1]

    def exploit(self, node):
        return node.value / node.visits

    def explore(self, node):
        return math.sqrt(2.0 * math.log(node.parent.visits) / node.visits)

    def best_child(self, node, factor = 1.0):
        """ Use upper confidence tree search to select a child node
        that returns the best scores.
	"""
        return max(node.children, key=lambda child: self.exploit(child) + factor * self.explore(child))
    
    def tree_policy(self, node):
        """ Implement expansion of the search tree. Return an unexplored node
        if the branch isn't fully explored, and the best child node if it is.
        """
        while not node.state.terminal_test():
            if not node.fully_explored():
                return self.expand(node)
            node = self.best_child(node)
        return node
        
    def rollout_policy(self, node, player):
        """ Copy the state of the latest expanded node, randomly choose 
        a legal move, play the game to a terminal state, and record a winner.
        """
        state = node.state
        while not state.terminal_test():
            action = random.choice(state.actions())
            state = state.result(action)
        if state._has_liberties(node.state.player()):
            return -1
        else: 
            return 1
    
    def backprop(self, node, value):
        """ Store the reward values of all the visited nodes while traveling
        backwards along the search tree. Flip the values between player 1 
        and player 2, to let the player tell its own moves from the opponent's.
        """
        while node:
            node.update(value)
            node = node.parent
            value = -value
    
    def monte_carlo_uct(self, state):
        """ Locate and backpropagate nodes along the search tree, corresponding 
        to moves which frequently lead to wins in rollout simulation games.
        """
        if 'TIME_LIMIT' in globals():
            # buffer = 50. # for slower but better performance
            buffer = 75. # for faster but decreased performance
            delta = (TIME_LIMIT-buffer) / 1000.
        else:
            # delta = 0.1 # for slower but better performance
            delta = 0.025 # for faster but decreased performance
        timer_end  = time.time() + delta
        root = Node(state)
        if root.state.terminal_test():
            return random.choice(state.actions())
        while time.time() < timer_end:
            leaf = self.tree_policy(root)
            value = self.rollout_policy(leaf, state.player())
            self.backprop(leaf, value)
        result = root.children.index(self.best_child(root, 0))
        return root.child_actions[result]

"""
    **********************************************************************
    CustomPlayer variable called by run_match.py (choose one of the above)
    **********************************************************************
"""

CustomPlayer = MonteCarloUCTPlayer
