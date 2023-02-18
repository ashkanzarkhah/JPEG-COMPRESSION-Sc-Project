import math
from PIL import Image
import numpy as np
from tabulate import tabulate
import os
from matplotlib import pyplot as PLT

#BULDING e
e = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 10, 103, 99]])

#BULDING C_8 and C_8 transpose
DCT = np.zeros((8,8))
for i in range(8):
    DCT[0][i] = 1 / math.sqrt(8)
for i in range(1, 8):
    for j in range(8):
        DCT[i][j] = 0.5 * math.cos((i * math.pi * (j + 0.5)) / 8)
DCTT = DCT.transpose()

#Inputing image
full_image = np.array(Image.open("Old.jpg"))

#Getting dimensions and making them divisible by 8
n = len(full_image)
n -= n % 8
m = len(full_image[0])
m -= m % 8

#Separating R and changing border from [0, 255] to [-127, 128]
red = np.zeros((n, m))
for i in range(n):
    for j in range(m):
        red[i][j] = full_image[i][j][0]
        red[i][j] -= 127

#Separating G and changing border from [0, 255] to [-127, 128]
green = np.zeros((n, m))
for i in range(n):
    for j in range(m):
        green[i][j] = full_image[i][j][1]
        green[i][j] -= 127

#Separating B and changing border from [0, 255] to [-127, 128]
blue = np.zeros((n, m))
for i in range(n):
    for j in range(m):
        blue[i][j] = full_image[i][j][2]
        blue[i][j] -= 127

#Transforming R
transformed_red = np.zeros((n, m))
for i in range(n // 8):
    for j in range(m // 8):
        bgi = i * 8; bgj = j * 8
        tmp = np.zeros((8,8))
        for ii in range(8):
            for jj in range(8):
                tmp[ii][jj] = red[bgi + ii][bgj + jj]
        tmp = np.dot(np.dot(DCT, tmp), DCTT)
        for ii in range(8):
            for jj in range(8):
                transformed_red[bgi + ii][bgj + jj] = tmp[ii][jj]
                
#Transforming G
transformed_green = np.zeros((n, m))
for i in range(n // 8):
    for j in range(m // 8):
        bgi = i * 8; bgj = j * 8
        tmp = np.zeros((8,8))
        for ii in range(8):
            for jj in range(8):
                tmp[ii][jj] = green[bgi + ii][bgj + jj]
        tmp = np.dot(np.dot(DCT, tmp), DCTT)
        for ii in range(8):
            for jj in range(8):
                transformed_green[bgi + ii][bgj + jj] = tmp[ii][jj]

#Transforming B
transformed_blue = np.zeros((n, m))
for i in range(n // 8):
    for j in range(m // 8):
        bgi = i * 8; bgj = j * 8
        tmp = np.zeros((8,8))
        for ii in range(8):
            for jj in range(8):
                tmp[ii][jj] = blue[bgi + ii][bgj + jj]
        tmp = np.dot(np.dot(DCT, tmp), DCTT)
        for ii in range(8):
            for jj in range(8):
                transformed_blue[bgi + ii][bgj + jj] = tmp[ii][jj]
                
#Quantizing R
quantized_red = np.zeros((n, m))
cnt_zero_red = 0
for i in range(n // 8):
    for j in range(m // 8):
        bgi = i * 8; bgj = j * 8
        tmp = np.zeros((8,8))
        for ii in range(8):
            for jj in range(8):
                tmp[ii][jj] = round(transformed_red[bgi + ii][bgj + jj] / e[ii][jj])
                cnt_zero_red += (tmp[ii][jj] == 0)
        for ii in range(8):
            for jj in range(8):
                quantized_red[bgi + ii][bgj + jj] = tmp[ii][jj]

#Quantizing G
quantized_green = np.zeros((n, m))
cnt_zero_green = 0
for i in range(n // 8):
    for j in range(m // 8):
        bgi = i * 8; bgj = j * 8
        tmp = np.zeros((8,8))
        for ii in range(8):
            for jj in range(8):
                tmp[ii][jj] = round(transformed_green[bgi + ii][bgj + jj] / e[ii][jj])
                cnt_zero_green += (tmp[ii][jj] == 0)
        for ii in range(8):
            for jj in range(8):
                quantized_green[bgi + ii][bgj + jj] = tmp[ii][jj]

#Quantizing B
quantized_blue = np.zeros((n, m))
cnt_zero_blue = 0
for i in range(n // 8):
    for j in range(m // 8):
        bgi = i * 8; bgj = j * 8
        tmp = np.zeros((8,8))
        for ii in range(8):
            for jj in range(8):
                tmp[ii][jj] = round(transformed_blue[bgi + ii][bgj + jj] / e[ii][jj])
                cnt_zero_blue += (tmp[ii][jj] == 0)
        for ii in range(8):
            for jj in range(8):
                quantized_blue[bgi + ii][bgj + jj] = tmp[ii][jj]

#Dequantizing R
new_transformed_red = np.zeros((n, m))
for i in range(n // 8):
    for j in range(m // 8):
        bgi = i * 8; bgj = j * 8
        tmp = np.zeros((8,8))
        for ii in range(8):
            for jj in range(8):
                tmp[ii][jj] = quantized_red[bgi + ii][bgj + jj]
        for ii in range(8):
            for jj in range(8):
                new_transformed_red[bgi + ii][bgj + jj] = tmp[ii][jj] * e[ii][jj]

#Dequantizing G
new_transformed_green = np.zeros((n, m))
for i in range(n // 8):
    for j in range(m // 8):
        bgi = i * 8; bgj = j * 8
        tmp = np.zeros((8,8))
        for ii in range(8):
            for jj in range(8):
                tmp[ii][jj] = quantized_green[bgi + ii][bgj + jj]
        for ii in range(8):
            for jj in range(8):
                new_transformed_green[bgi + ii][bgj + jj] = tmp[ii][jj] * e[ii][jj]
#Dequantizing B
new_transformed_blue = np.zeros((n, m))
for i in range(n // 8):
    for j in range(m // 8):
        bgi = i * 8; bgj = j * 8
        tmp = np.zeros((8,8))
        for ii in range(8):
            for jj in range(8):
                tmp[ii][jj] = quantized_blue[bgi + ii][bgj + jj]
        for ii in range(8):
            for jj in range(8):
                new_transformed_blue[bgi + ii][bgj + jj] = tmp[ii][jj] * e[ii][jj]

#Returning transformed red
new_red = np.zeros((n, m))
for i in range(n // 8):
    for j in range(m // 8):
        bgi = i * 8; bgj = j * 8
        tmp = np.zeros((8,8))
        for ii in range(8):
            for jj in range(8):
                tmp[ii][jj] = new_transformed_red[bgi + ii][bgj + jj]
        tmp = np.dot(np.dot(DCTT, tmp), DCT)
        for ii in range(8):
            for jj in range(8):
                new_red[bgi + ii][bgj + jj] = round(tmp[ii][jj] + 127)

#Returning transformed green
new_green = np.zeros((n, m))
for i in range(n // 8):
    for j in range(m // 8):
        bgi = i * 8; bgj = j * 8
        tmp = np.zeros((8,8))
        for ii in range(8):
            for jj in range(8):
                tmp[ii][jj] = new_transformed_green[bgi + ii][bgj + jj]
        tmp = np.dot(np.dot(DCTT, tmp), DCT)
        for ii in range(8):
            for jj in range(8):
                new_green[bgi + ii][bgj + jj] = round(tmp[ii][jj] + 127)

#Returning transformed red
new_blue = np.zeros((n, m))
for i in range(n // 8):
    for j in range(m // 8):
        bgi = i * 8; bgj = j * 8
        tmp = np.zeros((8,8))
        for ii in range(8):
            for jj in range(8):
                tmp[ii][jj] = new_transformed_blue[bgi + ii][bgj + jj]
        tmp = np.dot(np.dot(DCTT, tmp), DCT)
        for ii in range(8):
            for jj in range(8):
                new_blue[bgi + ii][bgj + jj] = round(tmp[ii][jj] + 127)


#Reconstructing full picture
new_full_image = [[[0,0,0] for i in range(m)] for j in range(n)]
for i in range(n):
    for j in range(m):
        new_full_image[i][j][0] = min(int(new_red[i][j]), 255)
        new_full_image[i][j][1] = min(int(new_green[i][j]), 255)
        new_full_image[i][j][2] = min(int(new_blue[i][j]), 255)

print("Mizan tonok bodane ghermez: ", (cnt_zero_red / (n * m)))
print("Mizan tonok bodane sabz: ", (cnt_zero_green / (n * m)))
print("Mizan tonok bodane abi: ", (cnt_zero_blue / (n * m)))
print("Mizane tonoki kol: ", ((cnt_zero_red + cnt_zero_green + cnt_zero_blue) / (3 * n * m)))

diff = 0
for i in range(n):
    for j in range(m):
        for k in range(3):
            diff += abs(full_image[i][j][k] - new_full_image[i][j][k]) * abs(full_image[i][j][k] - new_full_image[i][j][k])
total_diff = 0
for i in range(n):
    for j in range(m):
        for k in range(3):
            total_diff += abs(full_image[i][j][k]) * abs(full_image[i][j][k])
print("norm tafazole aks vorody vo khorogy:", math.sqrt(diff))
print("norm tafazole aks vorody vo khorogy bar roye norm vorody:", diff/total_diff)

PLT.imshow(new_full_image)
PLT.show()
