# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#
import random
import util

from game import Agent
from util import manhattanDistance


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood()  # food available from current state
        newFood = successorGameState.getFood()  # food available from successor state (excludes food@successor)
        currentCapsules = currentGameState.getCapsules()  # power pellets/capsules available from current state
        newCapsules = successorGameState.getCapsules()  # capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print (newPos, currentFood, newFood, currentCapsules, newCapsules, newGhostStates, newScaredTimes)

        newFoodList = newFood.asList()
        foodDistances = [1.0 / (manhattanDistance(newPos, food) + 1) for food in newFoodList]

        capsuleDistances = [1.0 / (manhattanDistance(newPos, capsule) + 1) for capsule in newCapsules]

        ghostDistances = [1.0 / (manhattanDistance(newPos, ghost.getPosition()) + 1) for ghost in newGhostStates]

        # Evaluation function score will be the sum of weighted features --> higher weight on food
        # and power capsules and negative on ghost proximity
        score = 10 * sum(foodDistances) + 4 * sum(capsuleDistances) + -10 * sum(ghostDistances)

        # Offset the score on stop action due to Pacman's timer
        if action == 'Stop':
            score -= 100

        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        def minimax(agentIndex, depth, gameState): # Recursive implementation:
            # agentIndex - the agent making the current decision
            # depth - depth of the current node in the search tree
            # gamestate - current state : isWin, isLose, getLegalActions, or generateSuccessor
            if depth == 0 or gameState.isWin() or gameState.isLose(): # is game at terminal state?
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(agentIndex) # what actions can the agent take at this position?
            successorStates = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
            # for each of the legal actions, generate the csuccessor states i.e. if the agent took the
            # legal action in legalActions, this is the state it would be in

            if agentIndex == 0:  # Agent 0 is pacman and so maximize score by choosing to ge to state that gives max score
                return max(minimax(1, depth, nextState) for nextState in successorStates)
                # we are calling agentIndex = 1 here because the recursive call is for the first ghost
                # recall the max level will be choosing from the minimizer node
            else:  # Current level is minimizing because its a ghost
                nextAgentIndex = agentIndex + 1
                if nextAgentIndex == gameState.getNumAgents():  # Move to the next depth (next ply)
                    # because all agents have made their move for this depth
                    nextAgentIndex = 0 #reset to Pacman playing
                    depth -= 1 # and go up the tree by one level
                return min(minimax(nextAgentIndex, depth, nextState) for nextState in successorStates)

        legalActions = gameState.getLegalActions(0)  # Pacman's legal actions
        scores = [minimax(1, self.depth, gameState.generateSuccessor(0, action)) for action in legalActions]
        bestScore = max(scores)
        bestActions = [action for action, score in zip(legalActions, scores) if score == bestScore]

        return random.choice(bestActions)  # Randomly choose among the best actions in case of a tie

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def alpha_beta_pruning(agentIndex, depth, gameState, alpha, beta):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(agentIndex)

            if agentIndex == 0:  # Max layer --> Pacman
                v = float('-inf')
                for action in legalActions:
                    nextState = gameState.generateSuccessor(agentIndex, action)
                    v = max(v, alpha_beta_pruning(1, depth - 1, nextState, alpha, beta))
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        break  # Prune remaining branches
                return v
            else:  # Min layer --> Ghosts
                v = float('inf')
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()  # Circular increment
                for action in legalActions:
                    nextState = gameState.generateSuccessor(agentIndex, action)
                    v = min(v, alpha_beta_pruning(nextAgentIndex, depth - (nextAgentIndex == 0), nextState, alpha, beta))
                    beta = min(beta, v)
                    if beta <= alpha:
                        break  # Prune remaining branches
                return v

        def alpha_beta_search():
            legalActions = gameState.getLegalActions(0)  # Pacman's legal actions
            bestScore = float('-inf')
            bestAction = None
            alpha = float('-inf')
            beta = float('inf')

            for action in legalActions:
                nextState = gameState.generateSuccessor(0, action)
                score = alpha_beta_pruning(1, self.depth, nextState, alpha, beta)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
                alpha = max(alpha, bestScore)
                if beta <= alpha:
                    break  # Prune remaining branches

            return bestAction

        return alpha_beta_search()



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(agentIndex, depth, gameState):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(agentIndex)
            successorStates = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]

            if agentIndex == 0:  # Max layer --> Pacman
                return max(expectimax(1, depth, nextState) for nextState in successorStates)
            else:  # Expectation layer --> Ghosts
                nextAgentIndex = agentIndex + 1
                if nextAgentIndex == gameState.getNumAgents():  # Move to the next depth (next ply)
                    nextAgentIndex = 0
                    depth -= 1
                expectedValue = sum(expectimax(nextAgentIndex, depth, nextState) for nextState in successorStates)
                return expectedValue / len(successorStates)

        legalActions = gameState.getLegalActions(0)  # Pacman's legal actions
        scores = [expectimax(1, self.depth, gameState.generateSuccessor(0, action)) for action in legalActions]
        bestScore = max(scores)
        bestActions = [action for action, score in zip(legalActions, scores) if score == bestScore]

        return random.choice(bestActions)  # Randomly choose among the best actions in case of a tie


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Combine important features in a linear combination with
    reciprocal values for better evaluation. Adjust weights based on the
    perceived importance of each feature.
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Minimize the reciprocal of the distance to the closest food
    foodDistances = [1.0 / (manhattanDistance(pacmanPos, food) + 1) for food in foodList.asList()]
    minFoodDistance = max(foodDistances) if foodDistances else 0

    # Minimize the reciprocal of the distance to the nearest ghost
    ghostDistances = [1.0 / (manhattanDistance(pacmanPos, ghost.getPosition()) + 1) for ghost in ghostStates]
    minGhostDistance = min(ghostDistances) if ghostDistances else 0

    # Minimize the reciprocal of the remaining number of capsules
    remainingCapsules = 1.0 / (len(capsules) + 1)  # Adding 1 to avoid division by zero

    # Score's based on the linear combination of features
    weights = [10, -8, 5]
    score = sum(weight * feature for weight, feature in zip(weights, [minFoodDistance, minGhostDistance, remainingCapsules]))

    return score + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
        Returns an action. You can use any method you want and search to any depth you want.
        Just remember that the mini-contest is timed, so you have to trade off speed and computation.

        Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
        just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        def evaluation_function(state):
            """
            Features considered in this contest agent:
            - closest food
            - closest scared ghost
            - closest non-scared ghost
            - remaining number of food pellets
            - closest power capsule

            The reciprocal values of the features are combined using a linear combination with weights to create a
            balanced evaluation that encourages Pac-Man to eat food, avoid ghosts, and use capsules strategically.
            """
            pacmanPos = state.getPacmanPosition()
            foodList = state.getFood()
            capsules = state.getCapsules()
            ghosts = state.getGhostStates()
            scaredTimes = [ghost.scaredTimer for ghost in ghosts]

            # Minimize the reciprocal of the distance to the closest food
            food_distances = [1.0 / (manhattanDistance(pacmanPos, food) + 1) for food in foodList.asList()]
            min_food_distance = max(food_distances) if food_distances else 0

            # Maximize the reciprocal of the distance to the closest scared ghost
            scared_ghost_distances = [1.0 / (manhattanDistance(pacmanPos, ghost.getPosition()) + 1)
                                      for ghost in ghosts if ghost.scaredTimer > 0]
            min_scared_ghost_distance = max(scared_ghost_distances) if scared_ghost_distances else 0

            # Minimize the reciprocal of the distance to the closest non-scared ghost
            non_scared_ghost_distances = [1.0 / (manhattanDistance(pacmanPos, ghost.getPosition()) + 1)
                                          for ghost in ghosts if ghost.scaredTimer == 0]
            min_non_scared_ghost_distance = max(non_scared_ghost_distances) if non_scared_ghost_distances else 0

            # Maximize the reciprocal of the remaining number of food pellets
            remaining_food_count = 1.0 / (foodList.count() + 1)

            # Minimize the reciprocal of the distance to the closest power capsule
            capsule_distances = [1.0 / (manhattanDistance(pacmanPos, capsule) + 1) for capsule in capsules]
            min_capsule_distance = max(capsule_distances) if capsule_distances else 0

            # Evaluation function score based on the linear combination of features
            weights = [3, -2, 2, -1, -1]
            score = sum(weight * feature for weight, feature in zip(weights,
                                                                    [min_food_distance, min_scared_ghost_distance,
                                                                     min_non_scared_ghost_distance, remaining_food_count,
                                                                     min_capsule_distance]))

            return score + state.getScore()

        def alpha_beta_pruning(agent_index, depth, state, alpha, beta):
            """
            Alpha-Beta Pruning algorithm. It prunes branches based on the evaluation function
            to reduce the search space and improve efficiency.
            """
            if depth == 0 or state.isWin() or state.isLose():
                return evaluation_function(state)

            legal_actions = state.getLegalActions(agent_index)
            successor_states = [state.generateSuccessor(agent_index, action) for action in legal_actions]

            if agent_index == 0:  # Max layer --> Pacman
                v = float('-inf')
                for next_state in successor_states:
                    v = max(v, alpha_beta_pruning(1, depth, next_state, alpha, beta))
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        break  # Prune remaining branches
                return v
            else:  # Min layer --> Ghosts
                v = float('inf')
                next_agent_index = agent_index + 1
                if next_agent_index == state.getNumAgents():  # Move to the next depth (next ply)
                    next_agent_index = 0
                    depth -= 1
                for next_state in successor_states:
                    v = min(v, alpha_beta_pruning(next_agent_index, depth, next_state, alpha, beta))
                    beta = min(beta, v)
                    if beta <= alpha:
                        break  # Prune remaining branches
                return v

        def alpha_beta_search():
            """
            Alpha-Beta search algorithm for finding the best action. It explores the game tree
            using the Alpha-Beta pruning algorithm and selects the best action based on the evaluation
            function.
            """
            legal_actions = gameState.getLegalActions(0)  # Pacman's legal actions
            best_score = float('-inf')
            best_action = None
            alpha = float('-inf')
            beta = float('inf')

            for action in legal_actions:
                successor_state = gameState.generateSuccessor(0, action)
                score = alpha_beta_pruning(1, self.depth, successor_state, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # Prune remaining branches

            return best_action

        return alpha_beta_search()
