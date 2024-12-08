# -*- coding: utf-8 -*-

import pygame
from pygame.locals import *
import numpy as np

print('Press p to pause/unpause. Add new cells in pause mode by right clicking.')

n = 21
cells = np.zeros((n,n),dtype=int)
cellsize = int(402/n)
pygame.init()
screen = pygame.display.set_mode((cellsize*n,cellsize*n))
cell = pygame.Surface((cellsize,cellsize))
cell.fill((255,255,255))
running = True
pause = True
stoptime = pygame.time.get_ticks()
while running:
    screen.fill((50,50,50))
    pygame.time.wait(5)
    if not pause:
        screen.fill((0,0,0))
        pygame.time.wait(250)
        neighbours = np.zeros((n,n),dtype=int)
        for r in range(n):
            for c in range(n):
                rmax = min(n,r+2)
                rmin = max(0,r-1)
                cmax = min(n,c+2)
                cmin = max(0,c-1)
                neighbours[r,c] = np.sum(cells[rmin:rmax,cmin:cmax])-cells[r,c]
        for r in range(n):
            for c in range(n):
                if neighbours[r,c] < 2:
                    cells[r,c] = 0
                elif neighbours[r,c] > 3:
                    cells[r,c] = 0
                elif cells[r,c]==0 and neighbours[r,c]==3:
                    cells[r,c] = 1

    for r in range(n):
        for c in range(n):
            if cells[r,c] == 1:
                screen.blit(cell,cell.get_rect(topleft=(r*cellsize,c*cellsize)))
    
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            elif event.key == K_p:
                pause = not pause    
        if event.type == pygame.MOUSEBUTTONDOWN:
            if pause:
            #Set the x, y postions of the mouse click
                x, y = event.pos
                cells[int(x/cellsize),int(y/cellsize)] = not cells[int(x/cellsize),int(y/cellsize)]
        elif event.type == QUIT:
            running = False
    pygame.display.flip()
    
pygame.quit()