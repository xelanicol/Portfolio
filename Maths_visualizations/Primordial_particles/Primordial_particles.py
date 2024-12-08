"""
Created on Sat Jun 26 10:08:43 2021

@author: alexa
"""

# -*- coding: utf-8 -*-

import pygame
from pygame.locals import *
import numpy as np
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
import os
import subprocess
from glob import glob

print('Press p to pause/unpause. Add new cells in pause mode by right clicking.')

w,h  = 600,400
n_particles = 150
z = 24
zoom = 0.5 # visual only
np.random.seed(42)
pos = np.stack([np.random.uniform( 0,z,n_particles),np.random.uniform(0,z,n_particles)]).T
#np.ones((n_particles,2))*np.array([[h/2,w/2]])
phi = np.random.uniform(0,2*np.pi,n_particles)
orient = np.stack([np.cos(phi),np.sin(phi)],axis=-1)
alpha = 180*np.pi/180
beta = 17*np.pi/180
neighbor_radius = 5
cm = plt.get_cmap('jet')
col = 255*np.ones((n_particles,3),dtype=np.int32)
v = 0.67
print('density: %.3f%%'%(100*n_particles/(z**2)))
def move(pos,orient):
    pos = pos + v*orient
    return pos

def rot_mat(a):
    return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])

def cosine_angle(a,b):
    dot = np.dot(a.T,b)
    return np.arccos( dot/(np.linalg.norm(a)+np.linalg.norm(b)) )


pygame.init()
screen = pygame.display.set_mode((w,h))
#cell = pygame.Surface((w,cellsize))
#cell.fill((255,255,255))
running = True
pause = True
stoptime = pygame.time.get_ticks()
t = 0
t0,t1,t2,t3, t4 = [datetime.now()]*5
it = 0
while running:
    #screen.fill((50,50,50))
    if not pause:
        screen.fill((0,0,0))
        #pygame.time.wait(5)
        pair_seps = (np.diff(np.array(list(itertools.permutations(pos,2))),axis=1)**2).sum(axis=-1).reshape(n_particles,-1)
        isneighbor = pair_seps < neighbor_radius**2
        idx = np.array(list(itertools.permutations(range(n_particles),2)))
        neighbor_pairs = idx[isneighbor.flatten()]
        neighbors = isneighbor.sum(axis=-1)
        col = 255*cm(np.minimum(np.ones_like(neighbors),np.log10(neighbors+1)/np.log10(0.2*n_particles)))[:,:3] #ignore alpha channel
        for i in range(n_particles):
            x = pos[i,1]
            y = pos[i,0]
            #if x > 0 and x < z and y > 0 and y < z:
            pygame.draw.circle(screen,col[i],(int(((1-zoom)/2)*w + x*zoom*w/z),int(((1-zoom)/2)*h + zoom*y*h/z)),int(w/150))
            #    if neighbors[i] > 0:
            #        pass
                    #pygame.draw.circle(screen,np.concatenate([col[i],[0.2]]),(int(x*w/z),int(y*h/z)),int(neighbor_radius*w/z),1)
        param_font = pygame.font.Font('freesansbold.ttf', w//50)
        param_text = param_font.render('max_iter = %d,  t1-t0: %.2f,  t2-t1: %.2f,  t3-t2: %.2f,  t4-t3: %.2f  t0-t4: %.2f'%(
                t,(t1-t0).microseconds/1000,(t2-t1).microseconds/1000,(t3-t2).microseconds/1000,(t4-t3).microseconds/1000,(t0-t4).microseconds/1000
            ), True, (255,255,255))
        screen.blit(param_text,param_text.get_rect(bottomleft=(10,h-10)))
        pygame.image.save(pygame.display.get_surface(), 'GIF/%05d.png'%(it))
        it += 1
        t0 = datetime.now()
        rotationpart = np.zeros_like(neighbors,dtype=np.float32)
        for (a,b) in neighbor_pairs:
            xprod =np.cross(orient[a],pos[b]-pos[a])
            R = xprod >= 0
            L = xprod < 0
            rotationpart[a] += int(R) - int(L)
        rotationpart[rotationpart > 0] = 1
        rotationpart[rotationpart < 0] = -1    
        t1 = datetime.now()
        dphi = alpha + beta*rotationpart*neighbors
        Rmat = np.moveaxis(np.array([[np.cos(dphi), np.sin(dphi)],[-np.sin(dphi),np.cos(dphi)]]),-1,0)
        t2 = datetime.now()
        orient= np.einsum('ijk,ikl->ijl',orient.reshape(n_particles,1,2),Rmat)[:,0,:]
        t3 = datetime.now()
        pos = move(pos,orient)
        t += 1
        t4 = datetime.now()
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            elif event.key == K_p:
                pause = not pause    
        elif event.type == QUIT:
            running = False
    pygame.display.flip()
pygame.quit()


subprocess.run([
    'ffmpeg',
    '-i',
    'GIF/%05d.png',
    'GIF/anim.gif','-y'
])
for f in glob('GIF/*.png'):
    os.remove(f)