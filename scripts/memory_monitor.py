import psutil
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import pickle

print(sys.argv)
if len(sys.argv) >= 2:
  pid = int(sys.argv[1])
else:
  pid = os.getpid()

process = psutil.Process(pid)

mem = []

start_t = time.time()

T = []
M = []

while True:
  t = time.time() - start_t
  time.sleep(0.05)
  T.append(t)
  M.append(process.memory_info().rss / (1024 ** 3))
  plt.clf()
  plt.plot(T, M)
  plt.xlabel('Time (seconds)')
  plt.ylabel('Memory Consumption (G Bytes)')
  plt.ylim(0, max(M) * 1.2)
  plt.draw()
  plt.pause(0.001)
  with open('mem.pkl', 'wb') as f:
    pickle.dump((T, M), f)

