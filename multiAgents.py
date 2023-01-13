"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Student name: Ben Levi
Student ID: 318811304

"""

# multiAgents.py
# --------------
# Attribution Information: part of the code were created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# http://ai.berkeley.edu.
# We thank them for that! :)


import random, util, math

from connect4 import Agent

import gameUtil as u


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 1 # agent is always index 1
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class BestRandom(MultiAgentSearchAgent):

    def getAction(self, gameState):

        return gameState.pick_best_move()


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def min_max(self, gameState, d):
        if d == 0 or gameState.is_terminal():
            return self.evaluationFunction(gameState)
        turn = gameState.turn
        children = gameState.getLegalActions()
        if turn == u.AI:
            cur_max = float('-inf')
            for c in children:
                s = gameState.generateSuccessor(self.index, c)
                s.switch_turn(gameState.turn)
                v = self.min_max(s, d - 1)
                cur_max = max(v, cur_max)
            return cur_max
        else:
            cur_min = float('inf')
            for c in children:
                s = gameState.generateSuccessor(self.index, c)
                s.switch_turn(gameState.turn)
                v = self.min_max(s, d - 1)
                cur_min = min(v, cur_min)
            return cur_min

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.isWin():
        Returns whether or not the game state is a winning state for the current turn player

        gameState.isLose():
        Returns whether or not the game state is a losing state for the current turn player

        gameState.is_terminal()
        Return whether or not that state is terminal
        """

        "*** YOUR CODE HERE ***"
        # Now gameState.turn = u.AI.
        options = gameState.getLegalActions()
        if not options:
            return None
        values = []
        for op in options:
            s = gameState.generateSuccessor(self.index, op)
            s.switch_turn(gameState.turn)
            values.append((self.min_max(s, self.depth - 1), op))
        # first arg max.
        return max(values, key=(lambda t: t[0]))[1]

        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):

    def max_value(self, gameState, a, b, d):
        if d == 0 or gameState.is_terminal():
            return [self.evaluationFunction(gameState), None]
        v = float('-inf')
        children = gameState.getLegalActions()
        op = None
        for c in children:
            s = gameState.generateSuccessor(self.index, c)
            s.switch_turn(gameState.turn)
            s_min_val = self.min_value(s, a, b, d-1)[0]
            if s_min_val > v:
                v = s_min_val
                op = c
            if v > b:
                return [v, op]
            a = max(a, v)
        return [v, op]

    def min_value(self, gameState, a, b, d):
        if d == 0 or gameState.is_terminal():
            return [self.evaluationFunction(gameState), None]
        v = float('inf')
        children = gameState.getLegalActions()
        op = None
        for c in children:
            s = gameState.generateSuccessor(self.index, c)
            s.switch_turn(gameState.turn)
            s_max_val = self.max_value(s, a, b, d - 1)[0]
            if s_max_val < v:
                v = s_max_val
                op = c

            v = min(v, self.max_value(s, a, b, d - 1)[0])
            if v < a:
                return [v, op]
            b = min(b, v)
        return [v, op]

    def getAction(self, gameState):
        """
            Your minimax agent with alpha-beta pruning (question 2)
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, float('-inf'), float('inf'), self.depth)[1]

        # util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def exp_val(self, gameState, d):
        v = 0
        children = gameState.getLegalActions()
        for c in children:
            s = gameState.generateSuccessor(self.index, c)
            s.switch_turn(gameState.turn)
            v += self.value_func(s, d - 1, True)[0]
        # uniform distribution.
        return [v / len(children), None]


    def max_val(self, gameState, d):
        v = float('-inf')
        children = gameState.getLegalActions()
        op = None
        for c in children:
            s = gameState.generateSuccessor(self.index, c)
            s.switch_turn(gameState.turn)
            s_val = self.value_func(s, d - 1, False)[0]
            if s_val > v:
                v = s_val
                op = c
        return [v, op]

    def value_func(self, gameState, d, is_max):
        if d == 0 or gameState.is_terminal():
            return [self.evaluationFunction(gameState), None]
        if is_max:
            return self.max_val(gameState, d)
        return self.exp_val(gameState, d)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value_func(gameState, self.depth, True)[1]

        # util.raiseNotDefined()
