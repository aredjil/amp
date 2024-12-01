#!/usr/bin/env python3
import numpy as np 
import matplotlib.pyplot as plt 
from amp.amp import amp
B = 4
n = 2**15
R =1.6
# Number of sections
L = int(n / B)

my_amp = amp(L, B)

snr = 15

ser_13 = my_amp.decode(snr, 1.3)
plt.plot(ser_13, label="R = 1.3", linestyle="dashed")
plt.yscale("log")
# plt.savefig(".  /plots/ser_r_16.png");
plt.show()