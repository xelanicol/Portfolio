"""
Created on Fri Oct  4 17:13:01 2019

@author: alexa
"""

import pygame
from pygame.locals import *
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
from glob import glob
import subprocess

for f in glob('GIF/*.png'):
    os.remove(f)

n = 200
cells = np.zeros((n,n),dtype=int)
cellsize = int(400/n)

class Ant:
    def __init__(self,pos,way):
        self.pos = pos
        self.way = way
        
def go_left(way):
    return (way-1)%4

def go_right(way):
    return (way+1)%4

n_ants = int(input('Please enter number of ants: '))
rule = input('Please enter pattern in the form of a string of Ls and Rs \n(e.g. LRRLLR)\n')
ants = []
vecs = []
for i in range(n_ants):
    ants.append(Ant(np.array([n/2,n/2],dtype=int),3))
    vecs.append(np.array([[1,0],[0,1],[-1,0],[0,-1]],dtype=int))

pygame.init()
screen = pygame.display.set_mode((cellsize*n,cellsize*n))
running = True
cells[tuple(ants[0].pos)] = 0

pause = False
stoptime = pygame.time.get_ticks()
it = 0

font = pygame.font.SysFont('calibri',12)
while running:
    screen.fill((200,200,200))
    if not pause:
        #screen.fill((255,255,255))
        #pygame.time.wait(500)
        for (ant,vec) in zip(ants,vecs):
            loc = tuple(ant.pos)
            if max(loc)>=n or min(loc)<0:
                running=False
                break
            if rule[cells[loc]] == 'L':
                ant.way = go_left(ant.way)
            elif rule[cells[loc]]== 'R':
                ant.way = go_right(ant.way)
            cells[loc] = (cells[loc]+1)%len(rule)
            ant.pos += vec[ant.way]
    
    image = np.zeros((n,n,4))#np.ones((n,n,4))
    image[cells>0,:] = cm.hsv(cells[cells>0]/(len(rule)-1))
    image = (255*image[:,:,:3]).astype(int)
    image = pygame.surfarray.make_surface(image.repeat(cellsize, axis=0).repeat(cellsize, axis=1))
    screen.blit(image,image.get_rect(topleft=(0,0)))
    text = font.render('iter: %d'%it,True,(200,200,200))
    textRect = text.get_rect(topleft=(10,10))
    screen.blit(text,textRect)
    
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[K_p]: 
                pause = not pause    
        if event.type == pygame.MOUSEBUTTONDOWN:
            if pause:
            #Set the x, y postions of the mouse click
                x, y = event.pos
                cells[int(x/cellsize),int(y/cellsize)] = not cells[int(x/cellsize),int(y/cellsize)]
        elif event.type == QUIT:
            running = False
    it += 1
    if it % 100 == 0:
        plt.imsave('GIF/%05d.png'%(it//100),pygame.surfarray.array3d(image))
    pygame.display.flip()
pygame.quit()
subprocess.run([
    'ffmpeg',
    '-i',
    'GIF/%05d.png',
    'GIF/Langtons Ant_%d_ants_%s.gif'%(n_ants,rule),
    '-y'
])

for f in glob('GIF/*.png'):
    os.remove(f)
plt.imsave('Langtons Ant_%d_ants_%s.png'%(n_ants,rule),pygame.surfarray.array3d(image))