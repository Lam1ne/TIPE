import chess
import numpy as np
import chess.pgn

def determiner_resultat(plateau):
    # Retourne 1 si les blancs gagnent, -1 si les noirs gagnent, 0 sinon
    if plateau.result() == "1-0":
        return 1
    elif plateau.result() == "0-1":
        return -1
    else:
        return 0

def index_piece(piece: chess.Piece):
    # Retourne un index unique pour chaque type de pièce
    piece_type = piece.piece_type - 1
    piece_color = 0 if piece.color == chess.WHITE else 6
    return piece_type + piece_color

def coups_index(coup: chess.Move, plateau: chess.Board, total_coups=4672):
    # Convertit un coup en un index unique
    from_square = coup.from_square
    to_square = coup.to_square
    promotion = coup.promotion

    index = from_square * 64 + to_square

    if promotion:
        # Ajuste l'index pour les promotions
        promotion_offset = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3
        }.get(promotion, 0)

        index += 64 * 64 + promotion_offset * 64 * 64

    index %= total_coups  # Assure que l'index est dans les limites

    return index


def coup_to_onehot(coup, plateau, total_coups=4672):
    # Crée un vecteur one-hot pour représenter le coup
    onehot = np.zeros(total_coups, dtype=np.float32)
    index = coups_index(coup, plateau)
    onehot[index] = 1
    return onehot

def enregistrer_partie_en_pgn(plateau, nom_fichier="partie.pgn"):
    # Enregistre la partie en format PGN
    game = chess.pgn.Game.from_board(plateau)
    game.headers["Event"] = "Partie d'auto-apprentissage"
    game.headers["White"] = "IA Blanc"
    game.headers["Black"] = "IA Noir"
    with open(nom_fichier, "w") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)

def transformer_etat(plateau: chess.Board):
    # Transforme l'état du plateau en une représentation utilisable par un réseau neuronal
    etat_transforme = np.zeros((8, 8, 112), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            piece = plateau.piece_at(chess.square(j, i))
            if piece:
                etat_transforme[i, j, index_piece(piece)] = 1
    historique_plateau = plateau.move_stack[-7:]
    plateau_copy = plateau.copy()
    for i in range(len(historique_plateau)-1, -1, -1):
        for j in range(8):
            for k in range(8):
                piece = plateau_copy.piece_at(chess.square(k, j))
                if piece:
                    etat_transforme[j, k, index_piece(piece) + (i+1)*12] = 1
        plateau_copy.pop()
    if plateau.has_kingside_castling_rights(chess.WHITE):
        etat_transforme[:, :, 104] = 1
    if plateau.has_queenside_castling_rights(chess.WHITE):
        etat_transforme[:, :, 105] = 1
    if plateau.has_kingside_castling_rights(chess.BLACK):
        etat_transforme[:, :, 106] = 1
    if plateau.has_queenside_castling_rights(chess.BLACK):
        etat_transforme[:, :, 107] = 1
    etat_transforme[:, :, 108] = 1 if plateau.turn == chess.BLACK else 0
    halfmove_clock = plateau.halfmove_clock
    fifty_move_rule_plan = halfmove_clock / 100.0
    etat_transforme[:, :, 110] = fifty_move_rule_plan
    etat_transforme[:, :, 111] = 1
    return etat_transforme

def calculate_policy_label(board_state, move_made, total_moves=4672):
    # Crée un label pour le coup effectué
    policy_label = np.zeros(total_moves, dtype=np.float32)
    move_index = coups_index(move_made, board_state)
    policy_label[move_index] = 1
    return policy_label

def calculate_value_label(outcome):
    # Mappe le résultat de la partie à une valeur
    if outcome == 1:  # Victoire
        return 1
    elif outcome == -1:  # Défaite
        return -1
    else:  # Nulle
        return 0