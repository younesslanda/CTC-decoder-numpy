# Author : Youness Landa

import numpy as np

from utils import pad_label
from alphabet import Alphabet

class CTCLayer:
    def __init__(self, label, outputs):
        self.label = label # label is l from the paper
        self.outputs = outputs # output : Y_{k}^{t}

    def forward(self):
        label = self.label
        outputs = self.outputs

        padded_label = pad_label(label)  # padded_label is l' from the paper. label is l from the paper
        num_time_steps = outputs.shape[0] # T
        padded_label_length = len(padded_label)
        last_padded_ind = padded_label_length - 1
        blank_label = Alphabet.blank_label

        # Alpha:
        alpha_table = np.zeros((num_time_steps, padded_label_length))

        def alpha(t, s):
            if s < 0 or s >= len(padded_label):
                return 0

            current_padded_character = padded_label[s]
            current_padded_label_score = outputs[t, Alphabet.alphabet_letter_to_ind[current_padded_character]]

            if t == 0:
                if s == 0:
                    return outputs[0, Alphabet.blank_ind]
                elif s == 1:
                    return current_padded_label_score
                else:
                    return 0

            # (6, 7) from the paper
            alpha_tag_t_s = alpha_table[t - 1, s] + (alpha_table[t - 1, s - 1] if s-1 >= 0 else 0)
            if current_padded_character == blank_label or (s >= 2 and padded_label[s-2] == current_padded_character):
                return alpha_tag_t_s * current_padded_label_score
            else:
                return (alpha_tag_t_s + (alpha_table[t - 1, s - 2] if s - 2 >= 0 else 0)) * current_padded_label_score

        for t in range(0, num_time_steps):
            for s in range(0, padded_label_length):
                alpha_table[t, s] = alpha(t, s)
        
        # Beta:
        beta_table = np.zeros((num_time_steps, padded_label_length))

        def beta(t, s):
            if s < 0 or s >= len(padded_label):
                return 0

            current_padded_character = padded_label[s]
            current_padded_label_score = outputs[t, Alphabet.alphabet_letter_to_ind[current_padded_character]]
            last_time_step = outputs.shape[0] - 1

            if t == last_time_step:
                if s == last_padded_ind:
                    return outputs[last_time_step, Alphabet.blank_ind]
                elif s == last_padded_ind - 1:
                    return current_padded_label_score
                else:
                    return 0
                    
            # (10, 11) from the paper.
            beta_tag_t_s = beta_table[t + 1, s] + (beta_table[t + 1, s + 1] if s + 1 <= last_padded_ind else 0)
            if current_padded_character == blank_label or \
                    (s + 2 <= last_padded_ind and padded_label[s+2] == current_padded_character):
                return beta_tag_t_s * current_padded_label_score
            else:
                return (beta_tag_t_s +
                        (beta_table[t + 1, s + 2] if s + 2 <= last_padded_ind else 0)) * current_padded_label_score

        for t in range(num_time_steps - 1, -1, -1):
            for s in range(padded_label_length - 1, -1, -1):
                beta_table[t, s] = beta(t, s)

        fwd_state = (alpha_table, beta_table)

        return fwd_state

    def backward(self, fwd_state):
        label = self.label
        outputs = self.outputs

        assert outputs.shape[0] >= len(label)

        alpha_table, beta_table = fwd_state

        padded_label = pad_label(label)
        gradients = np.zeros_like(outputs)

        score_last = alpha_table[ - 1, len(padded_label) - 1]
        score_before_last = alpha_table[outputs.shape[0] - 1, len(padded_label) - 2]
        p_l_given_ctc = score_last + score_before_last

        for t in range(outputs.shape[0]):
            for k in range(outputs.shape[1]):
                # Formula 15:
                d_p_d_ytk = 0
                lab_lk = np.nonzero(list(map(lambda x: 1 if Alphabet.alphabet_ind_to_letter[k] in x else 0, padded_label)))[0]
                for s in lab_lk:
                    d_p_d_ytk += alpha_table[t, s] * beta_table[t, s]

                d_p_d_ytk /= (outputs[t, k] ** 2)
                d_lnp_d_ytk = (1. / p_l_given_ctc) * d_p_d_ytk
                gradients[t, k] = d_lnp_d_ytk

        return gradients