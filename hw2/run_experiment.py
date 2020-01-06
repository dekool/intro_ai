from os import system
import sys

def run_exp(player1, depth=2):

    for i in range(50):
        command = f'python main.py --custom_game --p1 {player1} --p2 BetterGreedyAgent -f >> {player1}.{depth}'
        system(command)

if len(sys.argv) > 1:
    run_exp(sys.argv[1], sys.argv[2])

else:
    run_exp('TournamentAgent')