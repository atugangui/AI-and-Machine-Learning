from copy import deepcopy

__author__ = 'Team Oak'


def diagram_to_state(diagram):
    """Converts a list of strings into a list of lists of characters (strings of length 1.)"""
    return [list(a) for a in diagram]


INITIAL_STATE = diagram_to_state(['........',
                                  '........',
                                  '........',
                                  '...#O...',
                                  '...O#...',
                                  '........',
                                  '........',
                                  '........'])


def count_pieces(state):
    """Returns a dictionary of the counts of '#', 'O', and '.' in state."""
    count = {'#': 0, 'O': 0, '.': 0}
    for row in state:
        for square in row:
            count[square] += 1
    return count


def prettify(state):
    """
    Returns a single human-readable string representing state, including row and column indices and counts of
    each color.
    """
    string = ' 01234567\n'
    for row in range(8):
        string += str(row) + ''.join(state[row]) + str(row) + '\n'
    return string + ' 01234567\n' + str(count_pieces(state)) + '\n'


def opposite(color):
    """ opposite('#') returns 'O'. opposite('O') returns '#'. Assumes valid input."""
    return 'O' if color == '#' else '#'


def flips(state, r, c, color, dr, dc):
    """
    Returns a list of pieces that would be flipped if color played at r, c, but only searching along the line
    specified by dr and dc. For example, if dr is 1 and dc is -1, consider the line (r+1, c-1), (r+2, c-2), etc.

    :param state: The game state.
    :param r: The row of the piece to be  played.
    :param c: The column of the piece to be  played.
    :param color: The color that would play at r, c.
    :param dr: The amount to adjust r on each step along the line.
    :param dc: The amount to adjust c on each step along the line.
    :return A list of (r, c) pairs of pieces that would be flipped.
    """
    flipped = []
    for i in range(1, 100):
        if not (0 <= r + i * dr <= 7 and 0 <= c + i * dc <= 7):
            return
        if state[r + i * dr][c + i * dc] == opposite(color):
            flipped.append((r + i * dr, c + i * dc))
        elif state[r + i * dr][c + i * dc] == color:
            return flipped
        else:
            return


OFFSETS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def flips_something(state, r, c, color):
    """Returns True if color playing at r, c in state would flip something."""
    for offset in OFFSETS:
        if flips(state, r, c, color, offset[0], offset[1]):
            return True
    return False


def legal_moves(state, color):
    """
    Returns a list of legal moves (r, c) pairs) that color can make from state. Note that a player must flip
    something if possible; otherwise they must play the special move 'pass'.
    """
    moves = []
    for r in range(8):
        for c in range(8):
            if state[r][c] == '.' and flips_something(state, r, c, color):
                moves.append((r, c))
    return moves if moves else ['pass']


def successor(state, move, color):
    """
    Returns the state that would result from color playing move (which is either a pair (r, c) or 'pass').
    Assumes move is legal.
    """
    if move == 'pass':
        return state

    result = deepcopy(state)
    result[move[0]][move[1]] = color

    for offset in OFFSETS:
        flipped = flips(state, move[0], move[1], color, offset[0], offset[1])
        if flipped is None:
            continue
        for (r, c) in flipped:
            result[r][c] = opposite(result[r][c])

    return result


def score(state):
    """
    Returns the scores in state. More positive values (up to 64 for occupying the entire board) are better for '#'.
    More negative values (down to -64) are better for 'O'.
    """
    count = count_pieces(state)
    return count['#'] - count['O']


def game_over(state):
    """Returns true if neither player can flip anything."""
    return legal_moves(state, '#') == ['pass'] and legal_moves(state, 'O') == ['pass']
