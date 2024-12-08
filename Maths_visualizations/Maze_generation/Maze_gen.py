# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:26:06 2020

@author: alexa
"""

import numpy as np
import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import subprocess
from glob import glob
import os

for f in glob('GIF/*.png'):
    os.remove(f)


n= 41
grid = np.zeros((n,n),dtype=int)
grid[0,n//2] = 1
#grid[n-1,n//2] = 1
u = np.array([-1,0])
d = np.array([1,0])
l = np.array([0,-1])
r = np.array([0,1])
grid_revert = grid.copy()

pos = np.array([1,n//2])
end = np.array([n-2,n//2])
rot = np.array([[0,1],[-1,0]])

def dir2str(nparray):
    if np.array_equal(nparray,d):
        return 'd'
    elif np.array_equal(nparray,r):
        return 'r'
    elif np.array_equal(nparray,u):
        return 'u'
    elif np.array_equal(nparray,l):
        return 'l'
    else:
        return '??'


def is_valid(x,lim2=3):
    g = grid.copy()
    g[tuple(x)] = 1
    gidx =  np.argwhere(g[x[0]-1:x[0]+2,x[1]-1:x[1]+2]>0)+x-1
    lim = 3
    for x in list(gidx):
        ul = (g[x[0]-1:x[0]+1,  x[1]-1:x[1]+1] > 0).sum()
        ur = (g[x[0]-1:x[0]+1,  x[1]:x[1]+2] > 0).sum()
        dr = (g[x[0]:x[0]+2,    x[1]:x[1]+2] > 0).sum()
        dl = (g[x[0]:x[0]+2,    x[1]-1:x[1]+1] > 0).sum()
        if max([ul,ur,dl,dr]) > lim or sorted([ul,ur,dl,dr])[-3]>lim2:
    #ul < lim and ur < lim and dl < lim and dr < lim:
            return False
    return True

cellsize = 400//n
pygame.init()
screen = pygame.display.set_mode((cellsize*n,cellsize*n))
cell = pygame.Surface((cellsize,cellsize))
it = 0
route = []
nadj = []
prevpos = []
grid[tuple(pos)] = 1
running = True
pygame.time.wait(10000)
while not np.array_equal(pos,end) and running == True:
    it += 1
    screen.fill((0,0,0))
    adj = []
    
    if len(prevpos) > n**2:
        prevpos = []
    for v in [d,l,u,r]:
        if (pos+v).min() > 0 and (pos+v).max() < n-1:
            if grid[tuple(pos+v)] == 0:
                if not any([np.array_equal(prv,pos+v) for prv in prevpos]):
                    if is_valid(pos+v,lim2=2):
                        adj.append(pos+v)
    if len(adj)>0:
        nadj.append(len(adj))
        route.append(pos)
        pos = adj[np.random.randint(len(adj))]
        grid[tuple(pos)] = 1
    else:#find last route with multiple adj valid
        if not any([np.array_equal(prv,pos) for prv in prevpos]):
            prevpos.append(pos.copy())
        pos = route[-1]
        grid = grid_revert.copy()
        for cell in route:
            grid[tuple(cell)] = 1
        del route[-1]
        del nadj[-1]
    image = np.zeros((n,n,4))
    image[grid>0,:] = 1
    image[n-1,n//2] = 1
    image = (255*image[:,:,:3]).astype(int)
    image = pygame.surfarray.make_surface(image.repeat(cellsize, axis=0).repeat(cellsize, axis=1))
    screen.blit(image,image.get_rect(topleft=(0,0)))
    plt.imsave('GIF/%05d.png'%it,pygame.surfarray.array3d(image))
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False  
        elif event.type == QUIT:
            running = False
    pygame.display.flip() 
    pygame.time.wait(100)
grid[n-1,n//2] = 1
idx = np.argwhere(grid[1:n-1,1:n-1]>0)+1
valid_idx = idx[np.apply_along_axis(is_valid,1,idx)]
n_available = len(valid_idx)
n_remaining = n_available
pos = valid_idx[np.random.randint(len(valid_idx)),:]
grid_orig = grid.copy()
newchain = []
tried = []
while n_remaining/n_available > 0.1 and running == True:
    it += 1
    screen.fill((0,0,0))
    adj = []
    for v in [d,l,u,r]:
        if (pos+v).min() > 0 and (pos+v).max() < n-1:
            if grid[tuple(pos+v)] == 0:
                if grid[tuple(pos+v+d)] + grid[tuple(pos+v+u)] + grid[tuple(pos+v+r)] + grid[tuple(pos+v+l)] == 1:    
                    if is_valid(pos+v,lim2=2):
                        adj.append(pos+v)                    
    if len(adj)>0:
        pos = adj[np.random.randint(len(adj))]
        newchain.append(pos)
        grid[tuple(pos)] = 1
    else:
        tried.append(pos)
        if len(newchain) == 1:
            newchain = []
            grid[tuple(pos)] = 0
        newchain = []
        for coord in tried:
            grid[tuple(coord)] = 0
        idx = np.argwhere(grid[1:n-1,1:n-1]>0)+1
        for coord in tried:
            grid[tuple(coord)] = 1
        n_remaining = len(idx)
        if n_remaining>0:
            pos = idx[np.random.randint(len(idx)),:]
        else:
            break
    image = np.zeros((n,n,4))
    image[grid>0,:] = 1
    image[n-1,n//2] = 1
    image = (255*image[:,:,:3]).astype(int)
    image = pygame.surfarray.make_surface(image.repeat(cellsize, axis=0).repeat(cellsize, axis=1))
    screen.blit(image,image.get_rect(topleft=(0,0)))
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False  
        elif event.type == QUIT:
            running = False
    pygame.display.flip() 
    pygame.time.wait(100)
    plt.imsave('GIF/%05d.png'%it,pygame.surfarray.array3d(image))
while running == True:
    screen.fill((0,0,0))
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False  
        elif event.type == QUIT:
            running = False
    image = np.zeros((n,n,4))
    image[grid>0,:] = 1
    image[n-1,n//2] = 1
    image[grid==0,:] = 0.2
    image = (255*image[:,:,:3]).astype(int)
    image = pygame.surfarray.make_surface(image.repeat(cellsize, axis=0).repeat(cellsize, axis=1))
    screen.blit(image,image.get_rect(topleft=(0,0)))
    pygame.display.flip()

pygame.quit()

subprocess.run([
    'ffmpeg',
    '-i',
    'GIF/%05d.png',
    'GIF/mazegen.gif','-y'
])
for f in glob('GIF/*.png'):
    os.remove(f)

