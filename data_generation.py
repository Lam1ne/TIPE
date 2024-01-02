import numpy as np
import chess
from utils.chess_utils import determiner_resultat, coup_to_onehot, transformer_etat

def choisir_coup(plateau, politique):
    coups_possibles = list(plateau.legal_moves)
    index_coup_choisi = np.argmax(politique)
    coup_choisi = coups_possibles[index_coup_choisi]
    return coup_choisi

def jouer_partie_auto_generee(reseau):
    plateau = chess.Board()
    historique_etats = []
    historique_politiques = []
    historique_coups = []

    while not plateau.is_game_over():
        etat = transformer_etat(plateau)
        etat = np.expand_dims(etat, axis=0)
        politique = reseau.predict(etat)[0]
        coup = choisir_coup(plateau, politique)
        plateau.push(coup)

        historique_etats.append(etat)
        historique_politiques.append(politique)
        historique_coups.append(coup_to_onehot(coup, plateau))
    resultat = determiner_resultat(plateau)
    return historique_etats, historique_politiques, historique_coups, resultat

