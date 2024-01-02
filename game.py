import chess
from MCTS import mcts


class ChessSimulator:
    def __init__(self, reseau):
        print("Reseau Type before chesssimulator: ", type(reseau))  # Debug line
        self.plateau = chess.Board()
        self.reseau = reseau
        print("Reseau Type after chesssimulator: ", type(reseau))  # Debug line

    def jouer_coup_ia(self, itermax=100):
        coup_choisi = mcts(self.plateau, itermax, self.reseau)
        self.plateau.push(coup_choisi)

    def run(self):
        while not self.plateau.is_game_over():
            print(self.plateau)
            self.jouer_coup_ia(itermax=100)  # L'IA joue un coup

        print("Partie terminée:", self.plateau.result())
