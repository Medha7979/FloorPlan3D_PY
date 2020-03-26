# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:00:00 2019

@author: MEDHA
"""

from turtle import Turtle, Screen
import time
import itertools  
a=[1,1]
b=[2,3]
i=0
for x, y in zip(a, b):
    if(i<(len(a)-1)):
        x=a[i]
        y=b[i]
        x1=a[i+1]
        y1=b[i+1]
        if((x1>x)and(y1==y)):
            s="right"
        elif((x1<x)and(y1==y)):
            s="left"
        elif((x1==x)and(y1>y)):
            s="forward"
        elif((x1==x)and(y1<y)):
            s="backward"
        print(s)
        i+=1
    

NAME = "F"

BORDER = 1
BOX_WIDTH, BOX_HEIGHT = 6, 10  # letter bounding box
WIDTH, HEIGHT = BOX_WIDTH - BORDER * 2, BOX_HEIGHT - BORDER * 2  # letter size
time.sleep(12)

def letter_A(turtle):
    turtle.forward(HEIGHT / 2)
    for _ in range(3):
        turtle.forward(HEIGHT / 2)
        turtle.right(90)
        turtle.forward(WIDTH)
        turtle.right(90)
    turtle.forward(HEIGHT)

def letter_B(turtle):
    turtle.forward(HEIGHT /2)
    turtle.right(90)
    turtle.circle(-HEIGHT / 4, 180, 30)
    turtle.right(90)
    turtle.forward(HEIGHT )
    turtle.right(90)
    turtle.circle(-HEIGHT / 4, 180, 30)

def letter_C(turtle):
    turtle.left(90)
    turtle.circle(-HEIGHT/4, 180, 30)

def letter_D(turtle):
    turtle.forward(HEIGHT)
    turtle.right(90)
    turtle.circle(-HEIGHT / 2, 180, 30)

def letter_I(turtle):
    turtle.right(90)
    turtle.forward(WIDTH)
    turtle.backward(WIDTH / 2)
    turtle.left(90)
    turtle.forward(HEIGHT)
    turtle.right(90)
    turtle.backward(WIDTH / 2)
    turtle.forward(WIDTH)


def letter_F(turtle):
    turtle.right(90)
    turtle.forward(WIDTH/4)
    turtle.left(90)
    turtle.forward(HEIGHT/4)
    turtle.right(90)
    turtle.forward((3*WIDTH)/4)
    turtle.left(90)
    turtle.forward(HEIGHT/4)
    turtle.left(90)
    turtle.forward(WIDTH)
    turtle.right(90)
    turtle.backward(HEIGHT/2)
    turtle.forward(HEIGHT)
    turtle.right(90)
    turtle.forward(WIDTH/2)
    turtle.left(90)
    turtle.backward(WIDTH)
    turtle.left(90)
    turtle.backward(WIDTH/2)
    turtle.right(90)
    turtle.forward(WIDTH)
    turtle.right(90)
    turtle.backward(WIDTH/2)
    

def letter_N(turtle):
    turtle.forward(HEIGHT )
    turtle.goto(turtle.xcor() + WIDTH, BORDER)
    turtle.forward(HEIGHT)

def letter_G(turtle):
    turtle.forward(HEIGHT)
    turtle.right(90)
    turtle.forward(WIDTH/4)
    turtle.left(90)
    turtle.backward(WIDTH/2)
    turtle.right(90)
    turtle.forward(HEIGHT/4)
    turtle.left(90)
    turtle.forward(WIDTH/2)
    turtle.right(90)
    turtle.forward(WIDTH/4)
    turtle.right(90)
    turtle.forward(HEIGHT)
    turtle.right(90)
    turtle.circle(-HEIGHT/4 , 180, 30)
    turtle.left(90)
    turtle.backward(HEIGHT/2)
    turtle.left(90)
    turtle.forward((3*WIDTH)/4)
    turtle.right(90)
    turtle.forward(HEIGHT/4)
    turtle.left(90)
    turtle.forward(WIDTH/8)
    turtle.right(90)
    turtle.backward(WIDTH/2)


    
    
LETTERS = {'A': letter_A, 'B': letter_B, 'C':letter_C, 'D': letter_D, 'I': letter_I, 'N': letter_N, 'F':letter_F, 'G':letter_G}

wn = Screen()
wn.setup(800, 400)  # arbitrary
wn.title("Turtle writing my name: {}".format(NAME))
wn.setworldcoordinates(0, 0, BOX_WIDTH * len(NAME), BOX_HEIGHT)

marker = Turtle()

for counter, letter in enumerate(NAME):
    marker.penup()
    marker.goto(counter * BOX_WIDTH + BORDER, BORDER)
    marker.setheading(90)

    if letter in LETTERS:
        marker.pendown()
        LETTERS[letter](marker)

marker.hideturtle()

wn.mainloop()
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as ndimage
 
imageFile = '2d.jpg'
mat = cv2.imread(imageFile)
mat = mat[:,:,0] # get the first channel
rows, cols = mat.shape
xv, yv = np.meshgrid(range(cols), range(rows)[::-1])
 
blurred = ndimage.gaussian_filter(mat, sigma=(10, 10), order=0)
fig = plt.figure(figsize=(16,16))
 
ax = fig.add_subplot(221)
ax.imshow(mat, cmap='gray')
 
ax = fig.add_subplot(222, projection='3d')
ax.elev= 250
ax.plot_surface(xv, yv, mat)
 
ax = fig.add_subplot(223)
ax.imshow(blurred, cmap='gray')
 
ax = fig.add_subplot(224, projection='3d')
ax.elev= 400
ax.plot_surface(xv, yv, blurred)
plt.show()


