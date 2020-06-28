# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:12:09 2020

@author: dell
"""

import numpy as np

dice = 3
step = np.arange(dice)
for src in range(100):
    step += src

#%%
ladders = {1:11, 2:22, 3:33}
ladder_move = np.vectorize(lambda x: ladders[x] if x in ladders else x)
print(ladder_move)
ladder_move(1)
ladder_move(2)
ladder_move(3)
ladder_move(4)
print(ladder_move(np.arange(50)))
#%%
step2 = np.arange(90,110)
step22 = np.piecewise(step2, [step2 > 100, step2 <= 100],
    [lambda x: 200 - x, lambda x: x])
print(step2)
print(step22)