#!/usr/bin/env python3
import numpy as np 
import matplotlib.pyplot as plt 
import time 
from amp.amp import amp
#TODO: Take the snr, B, and L as a user input from the user 
#TODO: Add usage message.
B:int = 4
n:int = 2**15
R:float =1.6
# Number of sections
L:int = int(n / B)

my_amp = amp(L, B)

snr:int = 15
start:float = time.time()

ser_13:list[float] = my_amp.decode(snr, 1.3)
ser_14:list[float] = my_amp.decode(snr, 1.4)
ser_145:list[float] = my_amp.decode(snr, 1.45)
ser_16:list[float] = my_amp.decode(snr, 1.6)

end:float = time.time()

print(f"The program took: {(end -start)/60} Seconds")

plt.plot(ser_13, label="R = 1.3", linestyle="dashed")
plt.plot(ser_14, label="R = 1.4", linestyle="dashed")
plt.plot(ser_145, label="R = 1.45", linestyle="dashed")
plt.plot(ser_16, label="R = 1.6", linestyle="dashed")
plt.xlabel("#iterations")
plt.ylabel("Section Error Rate (SER)")
plt.title(f"Section error rate vs number of iterations B={B}, L ={L}, and snr={snr}")
plt.xlim(0, )
plt.yscale("log")
plt.legend(loc="best")
plt.grid(True)
plt.savefig("./ser.png")
plt.show()