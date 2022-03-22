# Author : Youness Landa

class Alphabet:
    """
        This is an example alphabet
    """

    blank_label   = '^'
    pure_alphabet = ['a', 'b', 'c', 'd']

    alphabet_letter_to_ind = {ch: ind for ind, ch in enumerate(pure_alphabet + [blank_label])}
    alphabet_ind_to_letter = {ind: ch for ind, ch in enumerate(pure_alphabet + [blank_label])}

    blank_ind = alphabet_letter_to_ind[blank_label]