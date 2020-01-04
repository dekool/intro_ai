import matplotlib
from matplotlib import pyplot as plt
from prettytable import PrettyTable

matplotlib.use("TkAgg")
ALL_DEPTHS = [0, 1, 2, 3, 4, 5]
NUM_LAYOUTS = 10

# Score(depth)
total_avg_score_reflex = [0, 0, 0, 0, 0, 0]
total_avg_score_better = [0, 0, 0, 0, 0, 0]
total_avg_score_minimax = [0, 0, 0, 0, 0, 0]
total_avg_score_alpha_beta = [0, 0, 0, 0, 0, 0]

# TurnTime(depth)
avg_turn_time_score_reflex = [0, 0, 0, 0, 0, 0]
avg_turn_time_score_better = [0, 0, 0, 0, 0, 0]
avg_turn_time_score_minimax = [0, 0, 0, 0, 0, 0]
avg_turn_time_score_alpha_beta = [0, 0, 0, 0, 0, 0]

# read input from file
with open("experiments.csv", 'r') as f:
    for line in f:

        splited_line = line.split(",")
        assert len(splited_line) == 4

        agent = splited_line[0]
        depth = int(splited_line[1])
        avg_score = float(splited_line[2])
        avg_turn_time = float(splited_line[3])

        # reflex
        if agent == 'GreedyAgent':
            total_avg_score_reflex[depth] += avg_score / NUM_LAYOUTS
            avg_turn_time_score_reflex[depth] += avg_turn_time / NUM_LAYOUTS

        # better
        if agent == 'BetterGreedyAgent':
            total_avg_score_better[depth] += avg_score / NUM_LAYOUTS
            avg_turn_time_score_better[depth] += avg_turn_time / NUM_LAYOUTS

        # minimax
        if agent == 'MinimaxAgent':
            total_avg_score_minimax[depth] += avg_score / NUM_LAYOUTS
            avg_turn_time_score_minimax[depth] += avg_turn_time / NUM_LAYOUTS

        # alpha_beta
        if agent == 'AlphaBetaAgent':
            total_avg_score_alpha_beta[depth] += avg_score / NUM_LAYOUTS
            avg_turn_time_score_alpha_beta[depth] += avg_turn_time / NUM_LAYOUTS

# ==============================================================================
#                               Score(depth)
# ==============================================================================

plt.plot(ALL_DEPTHS[1:2], total_avg_score_reflex[1:2], label='GreedyAgent',
         linestyle="-", marker="o")
plt.plot(ALL_DEPTHS[1:2], total_avg_score_better[1:2], label='BetterGreedyAgent',
         linestyle="-", marker="o")
plt.plot(ALL_DEPTHS[2:5], total_avg_score_minimax[2:5], label='MinimaxAgent',
         linestyle="-", marker="o")
plt.plot(ALL_DEPTHS[2:5], total_avg_score_alpha_beta[2:5], label='AlphaBetaAgent',
         linestyle="-", marker="o")

plt.xlabel("Depth")
plt.ylabel("Total Avg Score")
plt.title("TotalAvgScore(Depth)")
plt.legend()
plt.grid()
plt.show()

print("                                              Score Table")
t = PrettyTable(['Agent', 'depth=1', 'depth=2', 'depth=3', 'depth=4'])
t.add_row(['GreedyAgent', total_avg_score_reflex[1], '', '', ''])
t.add_row(['BetterGreedyAgent', total_avg_score_better[1], '', '', ''])
t.add_row(['MinimaxAgent', '', total_avg_score_minimax[2],
           total_avg_score_minimax[3], total_avg_score_minimax[4]])
t.add_row(['AlphaBetaAgent', '', total_avg_score_alpha_beta[2],
           total_avg_score_alpha_beta[3], total_avg_score_alpha_beta[4]])
print(t)

# ==============================================================================
#                               Time(depth)
# ==============================================================================

plt.plot(ALL_DEPTHS[1:2], avg_turn_time_score_reflex[1:2], label='GreedyAgent',
         linestyle="-", ms=10, marker="o")
plt.plot(ALL_DEPTHS[1:2], avg_turn_time_score_better[1:2], label='BetterGreedyAgent',
         linestyle="-", marker="o")
plt.plot(ALL_DEPTHS[2:5], avg_turn_time_score_minimax[2:5], label='MinimaxAgent',
         linestyle="-", marker="o")
plt.plot(ALL_DEPTHS[2:5], avg_turn_time_score_alpha_beta[2:5], label='AlphaBetaAgent',
         linestyle="-", marker="o")

plt.xlabel("Depth")
plt.ylabel("Avg Turn Time Score [sec]")
plt.title("AvgTurnTime(Depth)")
plt.legend()
plt.grid()
plt.show()

print("                                              Time Table")
t = PrettyTable(['Agent', 'depth=1', 'depth=2', 'depth=3', 'depth=4'])
t.add_row(['GreedyAgent', avg_turn_time_score_reflex[1], '', '', ''])
t.add_row(['BetterGreedyAgent', avg_turn_time_score_better[1], '', '', ''])
t.add_row(['MinimaxAgent', '', avg_turn_time_score_minimax[2],
           avg_turn_time_score_minimax[3], avg_turn_time_score_minimax[4]])
t.add_row(['AlphaBetaAgent', '', avg_turn_time_score_alpha_beta[2],
           avg_turn_time_score_alpha_beta[3], avg_turn_time_score_alpha_beta[4]])
print(t)

# #==============================================================================
# #               Agents preformence on minimaxClassic layout
# #==============================================================================

# print("Agents preformence on minimaxClassic layout")
# t = PrettyTable(['Agent', 'Score'])
# t.add_row(['MinMaxAgent', avg_score_minimaxClassic_depth_4_minimax])
# t.add_row(['AlphaBetaAgent', avg_score_minimaxClassic_depth_4_alpha_beta])
# t.add_row(['RandomExpectimaxAgent', avg_score_minimaxClassic_depth_4_random_expectimax])
# print (t)

# #==============================================================================
# #               Agents preformence on trappedClassic layout
# #==============================================================================

# print("Agents preformence on trappedClassic layout")
# t = PrettyTable(['Agent', 'Score'])
# t.add_row(['MinMaxAgent', avg_score_trappedClassic_depth_4_minimax])
# t.add_row(['AlphaBetaAgent', avg_score_trappedClassic_depth_4_alpha_beta])
# t.add_row(['RandomExpectimaxAgent', avg_score_trappedClassic_depth_4_random_expectimax])
# print (t)
