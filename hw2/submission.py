from environment import Player, GameState, GameAction, get_next_state, SnakeMovementDirections
from utils import get_fitness
import numpy as np
from enum import Enum


def nearest_fruit_distance(state: GameState, head_position: tuple) -> int:
    minimum = np.inf
    for fruit_location in state.fruits_locations:
        x_distance = np.abs(fruit_location[0] - head_position[0])
        y_distance = np.abs(fruit_location[1] - head_position[1])
        fruit_distance = x_distance + y_distance
        if fruit_distance < minimum:
            minimum = fruit_distance

    return minimum


def avg_distance_from_all_fruits(state: GameState, head_position: tuple) -> float:
    if len(state.fruits_locations) == 0:
        return 1
    dist = 0
    for fruit_location in state.fruits_locations:
        # distance from x location
        dist += np.abs(fruit_location[0] - head_position[0])
        # distance from y location
        dist += np.abs(fruit_location[1] - head_position[1])

    return dist / len(state.fruits_locations)

def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    if not state.snakes[player_index].alive:
        return state.snakes[player_index].length

    return 3*state.snakes[player_index].length +\
           2*(1/nearest_fruit_distance(state, state.snakes[player_index].head)) +\
           (1/avg_distance_from_all_fruits(state, state.snakes[player_index].head))


class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """
    DEPTH = 2

    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """
        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    def _utility(self, state: GameState):
        if state.current_winner[0] == self.player_index:
            return state.snakes[self.player_index].length
        else:
            return -1

    def _RB_minimax(self, tb_state: TurnBasedGameState, deciding_agent: Turn, d: int):
        if tb_state.game_state.is_terminal_state:
            return self._utility(tb_state.game_state)
        if d == 0:
            return heuristic(tb_state.game_state, self.player_index)
        if deciding_agent == self.Turn.AGENT_TURN:
            actions = tb_state.game_state.get_possible_actions(player_index=self.player_index)
            cur_max = -np.inf
            for action in actions:
                tb_state.agent_action = action
                value = self._RB_minimax(tb_state, self.Turn.OPPONENTS_TURN, d)
                cur_max = max(cur_max, value)
            return cur_max
        else:
            cur_min = np.inf
            for opponents_actions in tb_state.game_state.get_possible_actions_dicts_given_action(tb_state.agent_action,
                                                                                   player_index=self.player_index):
                next_state = get_next_state(tb_state.game_state, opponents_actions)
                new_tb_state = self.TurnBasedGameState(next_state, None)
                value = self._RB_minimax(new_tb_state, self.Turn.AGENT_TURN, d - 1)
                cur_min = min(cur_min, value)
            return cur_min

    def get_action(self, state: GameState) -> GameAction:
        game_state = self.TurnBasedGameState(state, None)
        actions = state.get_possible_actions(player_index=self.player_index)
        best_action = GameAction.STRAIGHT  # default action. will be changed
        best_value = -np.inf
        for action in actions:
            game_state.agent_action = action
            value = self._RB_minimax(game_state, self.Turn.OPPONENTS_TURN, self.DEPTH)
            if value > best_value:
                best_action = action
                best_value = value
        return best_action


class AlphaBetaAgent(MinimaxAgent):
    def _RB_alpha_beta(self, tb_state: MinimaxAgent.TurnBasedGameState, deciding_agent: MinimaxAgent.Turn, d: int, alpha: int, beta: int):
        if tb_state.game_state.is_terminal_state:
            return self._utility(tb_state.game_state)
        if d == 0:
            return heuristic(tb_state.game_state, self.player_index)
        if deciding_agent == self.Turn.AGENT_TURN:
            actions = tb_state.game_state.get_possible_actions(player_index=self.player_index)
            cur_max = -np.inf
            for action in actions:
                tb_state.agent_action = action
                value = self._RB_alpha_beta(tb_state, self.Turn.OPPONENTS_TURN, d, alpha, beta)
                cur_max = max(cur_max, value)
                alpha = max(cur_max, alpha)
                if cur_max >= beta:
                    return np.inf
            return cur_max
        else:
            cur_min = np.inf
            for opponents_actions in tb_state.game_state.get_possible_actions_dicts_given_action(tb_state.agent_action,
                                                                                   player_index=self.player_index):
                next_state = get_next_state(tb_state.game_state, opponents_actions)
                new_tb_state = self.TurnBasedGameState(next_state, None)
                value = self._RB_alpha_beta(new_tb_state, self.Turn.AGENT_TURN, d - 1, alpha, beta)
                cur_min = min(cur_min, value)
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return -np.inf
            return cur_min

    def get_action(self, state: GameState) -> GameAction:
        game_state = self.TurnBasedGameState(state, None)
        actions = state.get_possible_actions(player_index=self.player_index)
        best_action = GameAction.STRAIGHT  # default action. will be changed
        best_value = -np.inf
        for action in actions:
            game_state.agent_action = action
            value = self._RB_alpha_beta(game_state, self.Turn.OPPONENTS_TURN, self.DEPTH, -np.inf, np.inf)
            if value > best_value:
                best_action = action
                best_value = value
        return best_action


class LocalSearchState:
    def __init__(self, actions: list, game_duration):
        if len(actions) < game_duration:
            for i in range(game_duration - len(actions)):
                actions.append(actions[-1])
        self.actions = actions
        self._game_duration = game_duration

    def get_legal_actions_current_move(self, current_move_index):
        """
        this function gets a state of the local search (list of the snake moves) and the index of the current move to improve.
        the function returns all the new possible states after activating the legal local operators on the current move
        in the given state
        """
        for action in list(GameAction):
            # only if the new action is different from the current action state, than the operator is legal
            if action != self.actions[current_move_index]:
                new_actions_list = self.actions.copy()
                new_actions_list[current_move_index] = action
                yield LocalSearchState(new_actions_list, self._game_duration)

    def get_legal_actions_double_move(self, current_move_index):
        """
        this function gets a state of the local search (list of the snake moves).
        the function returns all the new possible states after activating the legal local operators on the current move
        and the next move in the given state
        """
        if current_move_index == self._game_duration - 1:
            return self.get_legal_actions_current_move(current_move_index)
        for current_action in list(GameAction):
            for next_action in list(GameAction):
                # only if the new action is different from one of current action or the next actions the operator is legal
                if current_action != self.actions[current_move_index] or next_action != self.actions[current_move_index+1]:
                    new_actions_list = self.actions.copy()
                    new_actions_list[current_move_index] = current_action
                    new_actions_list[current_move_index+1] = next_action
                    yield LocalSearchState(new_actions_list, self._game_duration)



def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    game_duration = 50
    initial_state = [GameAction.STRAIGHT]
    current_state = LocalSearchState(initial_state, game_duration)
    sideways = 0
    limit_sidesteps = 30
    for i in range(game_duration):
        best_val = -np.inf
        best_states = []
        for new_state in current_state.get_legal_actions_current_move(i):
            new_val = get_fitness(tuple(new_state.actions))
            if new_val > best_val:
                best_val = new_val
                best_states = [new_state]
            elif new_val == best_val:
                best_states.append(new_state)
        current_val = get_fitness(tuple(current_state.actions))
        if best_val > current_val:
            index = np.random.choice(len(best_states))
            current_state = best_states[index]
            print("best val so far:" + str(best_val))
            sideways = 0
        elif best_val == current_val and sideways < limit_sidesteps:
            # replace in random to one of the new best states, or stay with the current one
            best_states.append(current_state)
            index = np.random.choice(len(best_states))
            current_state = best_states[index]
            sideways += 1
        else:  # no more improving moves or no more sidesteps allowed
            pass
    best_val_found = get_fitness(tuple(current_state.actions))
    print("best move vector found: ")
    print(current_state.actions)
    print("best score found:")
    print(best_val_found)


def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    game_duration = 50
    initial_state = [GameAction.STRAIGHT]
    current_state = LocalSearchState(initial_state, game_duration)
    sideways = 0
    limit_sidesteps = 50
    last_best_val = -np.inf
    last_best_actions = initial_state
    random_steps_allowed = 3
    for r in range(random_steps_allowed):
        for i in range(game_duration):
            best_val = -np.inf
            best_states = []
            for new_state in current_state.get_legal_actions_double_move(i):
                new_val = get_fitness(tuple(new_state.actions))
                if new_val > best_val:
                    best_val = new_val
                    best_states = [new_state]
                elif new_val == best_val:
                    best_states.append(new_state)
            current_val = get_fitness(tuple(current_state.actions))
            if best_val > current_val:
                index = np.random.choice(len(best_states))
                current_state = best_states[index]
                print("best val so far:" + str(best_val))
                sideways = 0
            elif best_val == current_val and sideways < limit_sidesteps:
                best_states.append(current_state)
                index = np.random.choice(len(best_states))
                current_state = best_states[index]
                sideways += 1
            else:  # no more improving moves or no more sidesteps allowed
                pass
        current_val = get_fitness(tuple(current_state.actions))
        print("last loop over. his score is: " + str(current_val))
        print("Last loop vector: " + str(current_state.actions))
        if current_val > last_best_val: # the last run didn't improved at all
            last_best_val = current_val
            last_best_actions = current_state
            last_best_actions.actions.copy()
        # new random initial state
        for j in range(game_duration):
            current_state.actions[j] = np.random.choice(list(GameAction))
    print("best move vector found: ")
    print(last_best_actions.actions)
    print("best score found:")
    print(last_best_val)


class TournamentAgent(Player):

    def is_trap(self, state: GameState) -> bool:
        """
        this method checks whether the snake closing a loop with itself, meaning it might enter a death trap.
        the method checks it by checking if in front of the snake (in front in the direction of the snake) there is the
        snake's body
        """
        head = state.snakes[self.player_index].head
        direction = state.snakes[self.player_index].direction
        if direction == SnakeMovementDirections.UP:
            cell_to_check = (head[0] - 1, head[1])
        elif direction == SnakeMovementDirections.DOWN:
            cell_to_check = (head[0] + 1, head[1])
        elif direction == SnakeMovementDirections.RIGHT:
            cell_to_check = (head[0], head[1] + 1)
        else:
            cell_to_check = (head[0], head[1] - 1)
        # return True if in front of the snake's head is the snake's body
        return state.snakes[self.player_index].is_in_cell(cell_to_check)

    def trap_escape(self, state: GameState) -> GameAction:
        """
        this method will be called in case the snake entered a self-loop trap.
        the method will return which side is probably not entering the trap (it will never return STRAIGHT, because by
        definition of trap, STRAIGHT is the snake's body).
        the method first checks if only of LEFT or RIGHT means not instant death. in case both of them are available,
        the snake will prefer to go to the direction of it's tail - it will less likely be a death trap
        """
        head = state.snakes[self.player_index].head
        tail = state.snakes[self.player_index].tail_position
        direction = state.snakes[self.player_index].direction
        # cell_to_check saves the cell in front of the snake
        if direction == SnakeMovementDirections.UP:
            cell_to_check = (head[0] - 1, head[1])
        elif direction == SnakeMovementDirections.DOWN:
            cell_to_check = (head[0] + 1, head[1])
        elif direction == SnakeMovementDirections.RIGHT:
            cell_to_check = (head[0], head[1] + 1)
        else:
            cell_to_check = (head[0], head[1] - 1)

        if state.snakes[self.player_index].is_in_cell(cell_to_check):
            # if we reach here - in one of the sides there is a death trap. STRAIGHT is instant death
            if direction == SnakeMovementDirections.UP:
                # check if right is also death (if so - return left)
                if state.snakes[self.player_index].is_in_cell((head[0], head[1] + 1)):
                    return GameAction.LEFT
                # check if left is also death (if so - return right)
                elif state.snakes[self.player_index].is_in_cell((head[0], head[1] - 1)):
                    return GameAction.RIGHT
                # tail(x) > head(x) - the tail is to the right
                if tail[1] > head[1]:
                    return GameAction.RIGHT
                # the tail is to the left
                elif tail[1] < head[1]:
                    return GameAction.LEFT
                else:
                    return self.emergency(state, direction)
            elif direction == SnakeMovementDirections.DOWN:
                # check if right is also death (if so - return left)
                if state.snakes[self.player_index].is_in_cell((head[0], head[1] + 1)):
                    return GameAction.RIGHT
                # check if left is also death (if so - return right)
                elif state.snakes[self.player_index].is_in_cell((head[0], head[1] - 1)):
                    return GameAction.LEFT
                # tail(x) > head(x) - the tail is to the left
                if tail[1] > head[1]:
                    return GameAction.LEFT
                # the tail is to the right
                elif tail[1] < head[1]:
                    return GameAction.RIGHT
                else:
                    return self.emergency(state, direction)
            elif direction == SnakeMovementDirections.RIGHT:
                # check if up is also death (if so - return right, because in this direction, left==up)
                if state.snakes[self.player_index].is_in_cell((head[0] - 1, head[1])):
                    return GameAction.RIGHT
                # check if down is also death (if so - return left, because in this direction, right==down)
                elif state.snakes[self.player_index].is_in_cell((head[0] + 1, head[1])):
                    return GameAction.LEFT
                # tail(y) > head(y) - the tail is to the right
                if tail[0] > head[0]:
                    return GameAction.RIGHT
                # the tail is to the left
                elif tail[0] < head[0]:
                    return GameAction.LEFT
                else:
                    return self.emergency(state, direction)
            elif direction == SnakeMovementDirections.LEFT:
                # check if up is also death (if so - return left, because in this direction, right==up)
                if state.snakes[self.player_index].is_in_cell((head[0] - 1, head[1])):
                    return GameAction.LEFT
                # check if down is also death (if so - return right, because in this direction, left==down)
                elif state.snakes[self.player_index].is_in_cell((head[0] + 1, head[1])):
                    return GameAction.RIGHT
                # tail(y) > head(y) - the tail is to the left
                if tail[0] > head[0]:
                    return GameAction.LEFT
                # the tail is to the right
                elif tail[0] < head[0]:
                    return GameAction.RIGHT
                else:
                    return self.emergency(state, direction)

    def emergency(self, state: GameState, direction) -> GameAction:
        """
        this method will be called in case there is a death trap if the snake will turn to the right or to the left,
        and the tail is exactly behind the head of the snake
        in this case - we check if there is an empty cell in the right side of the tail or the left side, and turn to
        the side with out the empty cell (the direction the body of the snakes move
        if no cell is found also in this method - return just random... it is very rear to reach it
        """
        head = state.snakes[self.player_index].head
        tail = state.snakes[self.player_index].tail_position
        if direction == SnakeMovementDirections.UP:
            # check if the cell to the right of the tail is a snake body. if it is - turn RIGHT
            if state.snakes[self.player_index].is_in_cell((tail[0], tail[1] + 1)):
                return GameAction.RIGHT
            # check if the cell to the left of the tail is a snake body. if it is - turn LEFT
            if state.snakes[self.player_index].is_in_cell((tail[0], tail[1] - 1)):
                return GameAction.LEFT
        elif direction == SnakeMovementDirections.DOWN:
            # check if the cell to the right of the tail is a snake body. if it is - turn LEFT (because we are facing down)
            if state.snakes[self.player_index].is_in_cell((tail[0], tail[1] + 1)):
                return GameAction.LEFT
            # check if the cell to the left of the tail is a snake body. if it is - turn RIGHT (because we are facing down)
            if state.snakes[self.player_index].is_in_cell((tail[0], tail[1] - 1)):
                return GameAction.RIGHT
        elif direction == SnakeMovementDirections.RIGHT:
            # check if the cell below the tail is a snake body. if it is - turn RIGHT (because we are facing right)
            if state.snakes[self.player_index].is_in_cell((tail[0] + 1, tail[1])):
                return GameAction.RIGHT
            # check if the cell above the tail is a snake body. if it is - turn LEFT (because we are facing right)
            if state.snakes[self.player_index].is_in_cell((tail[0] - 1, tail[1])):
                return GameAction.LEFT
        else:
            # check if the cell below the tail is a snake body. if it is - turn LEFT (because we are facing left)
            if state.snakes[self.player_index].is_in_cell((tail[0] + 1, tail[1])):
                return GameAction.LEFT
            # check if the cell above the tail is a snake body. if it is - turn RIGHT (because we are facing left)
            if state.snakes[self.player_index].is_in_cell((tail[0] - 1, tail[1])):
                return GameAction.RIGHT
        # else - random...
        choice = np.random.choice(2)
        if choice == 0:
            return GameAction.LEFT
        return GameAction.RIGHT

    def fruit_rank(self, state: GameState, fruit_location: tuple) -> int:
        """
        calculate the number of fruits (including this one) which are at close distance
        from the given fruit
        """
        rank = 0
        for i in range(-3, 3):
            for j in range(-3, 3):
                if (fruit_location[0]+i, fruit_location[1]+j) in state.fruits_locations:
                    rank += 5 - (np.abs(i) + np.abs(j))
        return rank

    def fruits_ranks(self, state: GameState):
        fruits_ranks = {}
        for fruit_location in state.fruits_locations:
            fruits_ranks[fruit_location] = self.fruit_rank(state, fruit_location)
        return fruits_ranks

    def nearest_good_fruit_distance(self, state: GameState, head_position: tuple, fruits_ranks) -> int:
        minimum = np.inf
        for fruit_location in state.fruits_locations:
            x_distance = np.abs(fruit_location[0] - head_position[0])
            y_distance = np.abs(fruit_location[1] - head_position[1])
            fruit_distance = x_distance + y_distance
            if fruits_ranks[fruit_location] >= 10:
                if fruit_distance < minimum:
                    minimum = fruit_distance
        return minimum

    def tournament_heuristic(self, state: GameState) -> float:
        """
        Computes the heuristic value for the agent with player_index at the given state
        :param state:
        :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
        state.snakes array as well.
        :return:
        """
        if not state.snakes[self.player_index].alive:
            return state.snakes[self.player_index].length

        return 5.1 * state.snakes[self.player_index].length + \
               3 * (1 / nearest_fruit_distance(state, state.snakes[self.player_index].head)) + \
               5 * (1 / self.nearest_good_fruit_distance(state, state.snakes[self.player_index].head,self.fruits_ranks(state))) + \
               2 * (1 / avg_distance_from_all_fruits(state, state.snakes[self.player_index].head))

    def get_action(self, state: GameState) -> GameAction:
        if self.is_trap(state):
            return self.trap_escape(state)
        # init with all possible actions for the case where the agent is alone. it will (possibly) be overridden later
        best_actions = state.get_possible_actions(player_index=self.player_index)
        best_value = -np.inf
        for action in state.get_possible_actions(player_index=self.player_index):
            for opponents_actions in state.get_possible_actions_dicts_given_action(action,
                                                                                   player_index=self.player_index):
                opponents_actions[self.player_index] = action
                next_state = get_next_state(state, opponents_actions)
                h_value = self.tournament_heuristic(next_state)
                if h_value > best_value:
                    best_value = h_value
                    best_actions = [action]
                elif h_value == best_value:
                    best_actions.append(action)

        return np.random.choice(best_actions)


if __name__ == '__main__':
    SAHC_sideways()
    local_search()
