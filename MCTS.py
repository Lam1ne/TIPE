import math
import random
import numpy as np
import chess
from utils.chess_utils import transformer_etat, coups_index, determiner_resultat
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class NoeudMCTS:
    def __init__(self, plateau: chess.Board, coup=None, parent=None):
        self.plateau = plateau
        self.coup = coup
        self.parent = parent
        self.enfants = []
        self.victoires = 0
        self.visites = 0
        self.coups_non_essayes = list(plateau.legal_moves)
        self.etat = str(plateau)  # Add this line to store the state as a string


    def uct_selection_enfant(self):
        uct_k = math.sqrt(2)
        parent_visites = self.parent.visites if self.parent is not None else 1

        def uct_value(c):
            victoires_scalar = np.sum(c.victoires)
            # Adding a small amount of randomness to the UCT value
            return victoires_scalar / c.visites + uct_k * math.sqrt(2 * math.log(parent_visites) / (c.visites if c.visites > 0 else 1)) + random.uniform(0, 0.01)

        selectionne = max(self.enfants, key=uct_value)
        return selectionne

    def ajouter_enfant(self, m, b):
        nouveau_noeud = NoeudMCTS(plateau=b, coup=m, parent=self)
        self.coups_non_essayes.remove(m)
        self.enfants.append(nouveau_noeud)
        return nouveau_noeud

    def mettre_a_jour(self, resultat):
        self.visites += 1
        self.victoires += resultat

    def selectionner_coup_avec_reseau(self, reseau):
        etat = transformer_etat(self.plateau)  # Convertissez l'état du plateau pour le réseau
        etat = np.expand_dims(etat, axis=0)  # Ajoutez une dimension pour la prédiction

        politique, _ = reseau.predict(etat)  # Obtenez la politique du réseau
        politique = politique.flatten()  # Aplatir la sortie si nécessaire

        # Choisissez un coup basé sur la politique prédite
        coups_possibles = list(self.plateau.legal_moves)
        coups_scores = [(coup, politique[coups_index(coup, self.plateau)]) for coup in coups_possibles]
        coups_scores.sort(key=lambda x: x[1], reverse=True)  # Sort moves by score

        for coup, score in coups_scores:
            if coup in self.coups_non_essayes:
                return coup

        # If no untried moves are found in the policy, choose randomly
        if self.coups_non_essayes:
            selected_random_move = random.choice(self.coups_non_essayes)
            return selected_random_move

        # If there are no moves left, return None
        return None

    def simuler(self):
        temp_board = self.plateau.copy()
        while not temp_board.is_game_over():
            coup = random.choice(list(temp_board.legal_moves))
            temp_board.push(coup)
        return determiner_resultat(temp_board)

    def retropropagation(self, resultat, reseau):
        etat = transformer_etat(self.plateau)
        etat = np.expand_dims(etat, axis=0)
        _, valeur = reseau.predict(etat)

        self.mettre_a_jour(valeur)
        if self.parent:
            self.parent.retropropagation(resultat, reseau)

def mcts(etat_racine: chess.Board, itermax, reseau):
    transposition_table = {}  # Add this line to create the transposition table
    noeud_racine = NoeudMCTS(plateau=etat_racine)
    transposition_table[noeud_racine.etat] = noeud_racine  # Add this line to store the root node in the transposition table

    for iteration in range(itermax):
        noeud = noeud_racine
        etat = etat_racine.copy()

        # Sélection avec réseau
        while noeud.coups_non_essayes == [] and noeud.enfants != []:
            if noeud.plateau.is_game_over():
                break
            noeud = noeud.uct_selection_enfant()
            etat.push(noeud.coup)
            logging.debug(f"End of Iteration {iteration}, Selected Move: {noeud.coup.uci()}")

        # Expansion avec réseau
        if noeud.coups_non_essayes != []:
            coup = noeud.selectionner_coup_avec_reseau(reseau)
            etat.push(coup)
            if str(etat) in transposition_table:  # Add these lines to check if the state is in the transposition table
                noeud = transposition_table[str(etat)]
            else:
                noeud = noeud.ajouter_enfant(coup, etat)
                transposition_table[str(etat)] = noeud  # Add this line to store the new node in the transposition table


        # Simulation
        resultat = noeud.simuler()

        # Rétropropagation
        noeud.retropropagation(resultat, reseau)

    if noeud_racine.enfants:
        return max(noeud_racine.enfants, key=lambda c: np.sum(c.victoires) / c.visites).coup
    else:
        return None  # Retourner None ou choisir un coup aléatoire si aucun enfant n'est présent