from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from enum import Enum
# added imports
import time
#from main import start_custom_game

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


def fruit_rank(state: GameState, fruit: tuple) -> int:
    """
    calculate the number of fruits (including this one) which are at distance less than the square root of the board size
    from the given fruit
    """
    max_range = np.sqrt(np.min([state.board_size['width'],state.board_size['height']]))
    close_fruits = 0
    for fruit_location in state.fruits_locations:
        fruit_dist = 0
        # calculate the fruit distance
        fruit_dist += np.abs(fruit_location[0] - fruit[0])
        fruit_dist += np.abs(fruit_location[1] - fruit[1])
        if fruit_dist < max_range:
            close_fruits += 1
    return close_fruits


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
    DEPTH = 3

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
        best_action = None
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
                value = self._RB_minimax(tb_state, self.Turn.OPPONENTS_TURN, d, alpha, beta)
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
                value = self._RB_minimax(new_tb_state, self.Turn.AGENT_TURN, d - 1)
                cur_min = min(cur_min, value)
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return -np.inf
            return cur_min

    def get_action(self, state: GameState) -> GameAction:
        game_state = self.TurnBasedGameState(state, None)
        actions = state.get_possible_actions(player_index=self.player_index)
        best_action = None
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
                # only if the new action is different from the state action the operator is legal
                if current_action != self.actions[current_move_index] and next_action != self.actions[current_move_index+1]:
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
    initial_state = [GameAction.STRAIGHT] #, GameAction.STRAIGHT, GameAction.LEFT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.RIGHT,GameAction.STRAIGHT]
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
            current_state = np.random.choice(best_states)
            print("best val so far:" + str(best_val))
            sideways = 0
        elif best_val == current_val and sideways < limit_sidesteps:
            current_state = np.random.choice(best_states)
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
    #initial_state = [GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.LEFT, GameAction.STRAIGHT,
    #                 GameAction.STRAIGHT, GameAction.RIGHT, GameAction.STRAIGHT]
    initial_state = [GameAction.STRAIGHT]
    current_state = LocalSearchState(initial_state, game_duration)
    sideways = 0
    limit_sidesteps = 50
    last_best_val = -np.inf
    last_best_actions = initial_state
    random_steps_allowed = 5
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
                current_state = np.random.choice(best_states)
                print("best val so far:" + str(best_val))
                sideways = 0
            elif best_val == current_val and sideways < limit_sidesteps:
                current_state = np.random.choice(best_states)
                sideways += 1
            else:  # no more improving moves or no more sidesteps allowed
                pass
        current_val = get_fitness(tuple(current_state.actions))
        print("last loop over. his score is: " + str(current_val))
        if current_val > last_best_val: # the last run didn't improved at all
            last_best_val = current_val
            last_best_actions = current_state
        # new random initial state
        for j in range(game_duration):
            current_state.actions[j] = np.random.choice(list(GameAction))
    print("best move vector found: ")
    print(last_best_actions.actions)
    print("best score found:")
    print(last_best_val)



# TODO: delete this function - it is only for testing!
from agents import GreedyAgent, StaticAgent, RandomPlayer
from environment import SnakesBackendSync, Grid2DSize
def get_fitness2(moves_sequence: tuple) -> float:
    n_agents = 20
    static_agent = StaticAgent(moves_sequence)
    opponents = [RandomPlayer() for _ in range(n_agents - 1)]
    players = [static_agent] + opponents

    board_width = 40
    board_height = 40
    n_fruits = 50
    game_duration = len(moves_sequence)

    env = SnakesBackendSync(players,
                            grid_size=Grid2DSize(board_width, board_height),
                            n_fruits=n_fruits,
                            game_duration_in_turns=game_duration, random_seed=42)
    env.run_game(human_speed=True, render=True)
    np.random.seed()
    return env.game_state.snakes[0].length + env.game_state.snakes[0].alive


def run_experiments():
    agents = ["GreedyAgent", "BetterGreedyAgent", "MinimaxAgent", "AlphaBetaAgent"]
    depths = [2, 3, 4]
    default_duration = 500
    default_width = 50
    default_height = 50
    default_fruits = 51
    for agent in agents:
        if agent in ["GreedyAgent", "BetterGreedyAgent"]:
            start_custom_game(agent, "GreedyAgent")
        else:
            for d in depths:
                pass

class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        pass


if __name__ == '__main__':
    #get_fitness2(tuple([GameAction.LEFT,GameAction.LEFT,GameAction.LEFT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.LEFT,GameAction.STRAIGHT,GameAction.LEFT,GameAction.LEFT,GameAction.STRAIGHT,GameAction.RIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.RIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.LEFT,GameAction.LEFT,GameAction.RIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.STRAIGHT,GameAction.LEFT,GameAction.LEFT,GameAction.STRAIGHT,GameAction.RIGHT,GameAction.RIGHT,GameAction.STRAIGHT]))
    #get_fitness2(tuple([GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.LEFT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.RIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.LEFT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.LEFT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.LEFT, GameAction.LEFT]))
    #SAHC_sideways()
    local_search()
    #run_experiments()
