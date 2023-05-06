import numpy as np

# I am cat
# Vectors of different words from embeddings
v_i = np.array([0.996,-0.23,0.556]) #1
v_am = np.array([0.696,-0.2,0.986]) #2
v_cat = np.array([0.86,0.4,0.16]) #3

# 3 dim vector of i am cat

v = [v_i,v_am,v_cat]

print("Vector:",v)
w_11 = v_i*v_i
w_12 = v_i*v_am
w_13 = v_i*v_cat

print(f"\nw11:{w_11}\nw12:{w_12}\nw13:{w_13}\n")

y1 = w_11*v_i + w_12*v_am + w_13*v_cat

print(f"y_1:{y1}")