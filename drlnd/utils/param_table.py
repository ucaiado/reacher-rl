#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
structure a statsmodel table


@author: udacity, ucaiado

Created on 10/07/2018
"""

import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.compat.python import zip_longest
from statsmodels.iolib.tableformatting import fmt_2cols


def convert_table_to_dict(table):
    '''
    convert statsmodel table to dict

    :param table: StatsModel table. Parameters of the agent
    '''
    l_aux = [y.strip() for y in np.array(table.data).flatten()]
    return dict([l_aux[y], float(l_aux[y+1])] for y in range(0, len(l_aux), 2))


def generate_table(left_col, right_col, table_title):
    # Do not use column headers
    col_headers = None

    # Generate the right table
    if right_col:
        # Add padding
        if len(right_col) < len(left_col):
            right_col += [(' ', ' ')] * (len(left_col) - len(right_col))
        elif len(right_col) > len(left_col):
            left_col += [(' ', ' ')] * (len(right_col) - len(left_col))
        right_col = [('%-21s' % ('  '+k), v) for k, v in right_col]

        # Generate the right table
        gen_stubs_right, gen_data_right = zip_longest(*right_col)
        gen_table_right = SimpleTable(gen_data_right,
                                      col_headers,
                                      gen_stubs_right,
                                      title=table_title,
                                      txt_fmt=fmt_2cols)
    else:
        # If there is no right table set the right table to empty
        gen_table_right = []

    # Generate the left table
    gen_stubs_left, gen_data_left = zip_longest(*left_col)
    gen_table_left = SimpleTable(gen_data_left,
                                 col_headers,
                                 gen_stubs_left,
                                 title=table_title,
                                 txt_fmt=fmt_2cols)

    # Merge the left and right tables to make a single table
    gen_table_left.extend_right(gen_table_right)
    general_table = gen_table_left

    return general_table
