# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 17:51:32 2018

@author: Patrick
"""


import matplotlib.pyplot as plt

years = range(2000, 2016)



errors = (6.52,7.11,2.36,1.37,15.37,2.02, 2.92, 7.20, 2.20, 12.53, 2.16, 31.68, 4.54, 17.21, 1.76, 10.81)

plt.scatter(years, errors)
plt.show()

plt.plot(years, errors)

plt.show()