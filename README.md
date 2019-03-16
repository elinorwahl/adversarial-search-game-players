# Adversarial Search Game-Playing Agent

### Synopsis

In this project, I have built two agents that use adversarial search techniques - minimax with alpha-beta search pruning, and Monte Carlo tree search - to play knight's Isolation. This is adapted from Udacity's Artificial Intelligence course materials, which can be found [here](https://github.com/udacity/artificial-intelligence/tree/master/Projects/3_Adversarial%20Search).

![Example game of isolation on a square board](viz.gif)

In the game Isolation, two players each control one token, and take turns moving their token from one cell to another on a rectangular grid. In knight's Isolation, tokens can move to any open cell that is 2-rows-and-1-column or 2-columns-and-1-row away from their current position on the board, in an 'L' shape. Whenever a token occupies a cell, that cell is blocked for the remainder of the game. On a blank board, this means that tokens have, at most, eight open cells to move into surrounding their current location. 'Knight' tokens can 'jump' blocked or occupied spaces (just like a knight in chess). The first player with no remaining open moves, or 'liberties' for their token loses the game, and their opponent is declared the winner.

In this implementation, agents have a fixed time limit (150 milliseconds by default) to search for the best move and respond. The search is automatically cut off after the time limit expires, and the active agent will forfeit the game if it has not chosen a move.

### Search Techniques

`sample_players.py` contains three players to use as opponents - 'Random', 'Greedy', and 'Minimax'. In `my_custom_player.py`, I have implemented two more advanced agents: Minimax with alpha-beta search pruning, and Monte Carlo tree search (MCTS) with upper confidence trees. This is how each of them function:

- `RandomPlayer` randomly selects a move from the moves available to its position on the board at each turn.

- `GreedyPlayer` selects a move projected to maximize its own score at each turn.

- `MinimaxPlayer` functions by 'maximizing' the 'minimum' available reward - it calculates the smallest possible reward that it can be forced into by another player's move, and the largest possible reward it can obtain when it knows the actions of the opposing player.

- `MinimaxAlphaBetaPlayer` improves on the minimax algorithm by employing an 'alpha' value representing the minimum score that the maximizing player can obtain, and a 'beta' value representing the maximum score that the minimizing player can obtain.

- `MonteCarloUCTPlayer` acts in four stages: 'Selection', in which it chooses a node and travels along a branch, 'expansion', in which it creates more nodes if a terminal state hasn't been reached and selects the most promising child node, 'simulation', in which it runs a simulated playout from that node and returns the final result, and 'backpropagation', in which it applies the result of the simulation moves to all the previous nodes along the current search tree. The 'upper confidence trees' function is provided by an equation within the selection function, which combines ‘exploration' of new paths with ‘exploitation' of paths that result in high scores for the player. Exploration is disabled outside of the simulation phase (the 'factor' that serves as multiplier for exploration is set to 0), so in actual gameplay the agent uses only the moves that it knows are likely to result in a victory.

### Performance

To test the Minimax Alpha-Beta Player and the Monte Carlo UCT Player, I used each to play 100 rounds of Isolation against the Random, Greedy, and Minimax players. This table represents the percentage of matches they won against each opponent:

|| Against Random Agent | Against Greedy Agent | Against Minimax Agent |
|:-:|:-:|:-:|:-:|
| Minimax Alpha-Beta Agent | 95% | 86% | 69% |
| Monte Carlo UCT Agent | 100% | 94% | 89% |

These are both markedly better than even an agent playing by the minimax heuristic!

- The minimax alpha-beta agent performs better than minimax because of increased efficiency, which makes it run faster. When, at a particular node, the maximum score that the 'beta' player can obtain becomes less than the minimum score the 'alpha' player can obtain, the children of this node are dropped from consideration while traveling the search tree, and the game moves they represent will no longer be attempted. This pruning of the search tree makes the agent's processes much faster and more efficient.

- The Monte Carlo agent performs better than minimax because of increased calculation power, with fewer consequences on active gameplay. The exploit/explore equation provides a stronger mathematical basis for choosing potential moves than either minimax or minimax alpha-beta, and the simulated rollout function ensures that the agent can test the effectiveness of moves without endangering its position while the game is in progress. Then, backpropagation lets the the algorithm distribute the results backwards along the entire search path, further cementing the learned information about the value of potential moves.

However - the Monte Carlo agent takes considerably longer to play matches than the minimax alpha-beta agent. The win rates recorded above only apply to the original version of the `monte_carlo_uct` function, which is this:

```
    def monte_carlo_uct(self, state):
        if 'TIME_LIMIT' in globals():
            buffer = 50.
            delta = (TIME_LIMIT-buffer) / 1000.
        else:
            # delta = 0.1
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
```

This agent, while extremely successful at Isolation, runs slowly enough that it sometimes can't make a move within the allotted time limit of 150 milliseconds. The version of the function that can meet the time limit is this:

``` 
    def monte_carlo_uct(self, state):
        if 'TIME_LIMIT' in globals():
            buffer = 75.
            delta = (TIME_LIMIT-buffer) / 1000.
        else:
            delta = 0.025
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
```

This simple change makes the Monte Carlo tree search run much faster, but with a serious impact on performance. With this in effect, the Monte Carlo agent won 71% of 100 matches against the minimax agent. Furthermore, running it often resulted in a hung terminal. Where the Monte Carlo algorithm is concerned, there appears to be a very large trade-off between speed and performance.

So how to fix this? One possible answer: Use NumPy instead of the built-in `math` package. While NumPy can't function with `isolation.py` for use in this project, it would make the program run much more quickly than the 'math' package by enabling it to perform calculations across entire vectors of numbers. This is critical because in order to be effective, Monte Carlo tree search must explore and test a large number of nodes.

### References

A very good, simple implementation of Monte Carlo tree search, which I used to build my own program, can be found on GitHub in [this Connect 4 player](https://github.com/Alfo5123/Connect4).

Additionally, [this blog post on Monte Carlo UCT search](http://www.moderndescartes.com/essays/deep_dive_mcts/) and [the associated code](https://github.com/brilee/python_uct) are excellent explanations of how Monte Carlo tree search with an upper confidence heuristic works, and how to optimize its functioning.
