import pygame
from pygame.locals import *
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import subprocess
import os
from glob import glob

def mandelbrot_recurs(carr, zarr = None, narr = None, max_iter = 100):
    if zarr is None:
        zarr = np.zeros_like(carr)
        narr = np.zeros(carr.shape,dtype='int')
    idx = np.logical_and(abs(zarr) < 2,narr < max_iter) # cells to keep iterating
    if np.sum(idx)==0: # cells that are finished
        zarr = zarr**2 + carr
        narr = narr + 1
        zarr = zarr**2 + carr
        narr = narr + 1
        result = narr
        # smoothing
        result = narr.astype(np.float32)
        result[narr<max_iter] = narr[narr<max_iter] + 1.0 - np.log(np.log2(abs(zarr[narr<max_iter])))
        # histogram coloring
        counts, bins = np.histogram(result[narr<max_iter],bins=max_iter*50)
        result[narr<max_iter] = (np.cumsum(counts)/counts.sum())[np.digitize(result[narr<max_iter],bins[1:],True)]
        #normalise result
        result[narr<max_iter] = (result[narr<max_iter] - result[narr<max_iter].min()) / (result[narr<max_iter].max()-result[narr<max_iter].min())
        result[result>=max_iter] = 0.0
        return result
    else:
        zarr[idx] = zarr[idx]**2 + carr[idx]
        narr[idx] = narr[idx] + 1
        return mandelbrot_recurs(carr, zarr, narr, max_iter)

nx, ny = 300,300
aspect = nx/ny

pygame.init()
screen = pygame.display.set_mode((nx,ny))
pygame.display.set_caption('Mandelbrot Set - right click to zoom')

xc, yc = -0.5,0
zoom = 1.5
zoom_step_factor = 0.5
c = 256
cmap = cm.inferno
x = np.sum(np.meshgrid(np.linspace(xc-zoom*aspect,xc+zoom*aspect,nx),np.linspace(yc-zoom,yc+zoom,ny)*1j),axis=0)
y= mandelbrot_recurs(x,max_iter=c)
y1 = cmap(y)
load_font = pygame.font.Font('freesansbold.ttf', nx//10)
load_text = load_font.render( 'L O A D I N G', True, (255,255,255),)
running = True
stoptime = pygame.time.get_ticks()
do_zoom = False
i = 0
plt.imsave('%03d.png'%i,y1)
surf = pygame.surfarray.make_surface(np.swapaxes(y1[:,:,:-1],0,1)*255)
screen.blit(surf,surf.get_rect( topleft=(0,0)))
while running:
    if do_zoom:
        param_font = pygame.font.Font('freesansbold.ttf', nx//20)
        param_text = param_font.render('max_iter = %d'%c, True, (255,255,255),)
        screen.blit(load_text,load_text.get_rect(center=(nx//2,ny//2)))
        screen.blit(param_text,param_text.get_rect(center=(nx//2,ny//2 - nx//10)))
        pygame.draw.rect(screen,(255,255,255),
                             zoomrect,
                             2)
        pygame.display.flip()
        xc, yc = x,y
        zoom *= zoom_step_factor
        x = np.sum(np.meshgrid(np.linspace(xc-zoom*aspect,xc+zoom*aspect,nx),np.linspace(yc-zoom,yc+zoom,ny)*1j),axis=0)
        y= mandelbrot_recurs(x,max_iter=c)
        y1 = cmap(y)
        i += 1

        plt.imsave('GIF/%03d.png'%i,y1)
        surf = pygame.surfarray.make_surface(np.swapaxes(y1[:,:,:-1],0,1)*255)
        screen.blit(surf,surf.get_rect(topleft=(0,0)))
        do_zoom = False
    pygame.time.wait(5)
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False  
        if event.type == pygame.MOUSEBUTTONDOWN:
            #Set the x, y postions of the mouse click
            x,y = event.pos
            zoomrect = pygame.Rect(0,0,int(nx*zoom_step_factor*aspect),int(ny*zoom_step_factor))
            zoomrect.center=(x,y)
            y = yc-zoom+((y+1)/ny)*zoom*2
            x = xc-zoom*aspect+((x+1)/nx)*zoom*aspect*2
            do_zoom = True
        elif event.type == QUIT:
            running = False
    pygame.display.flip()
pygame.quit()
subprocess.run([
    'ffmpeg',
    '-framerate','3',
    '-i',
    'GIF/%03d.png',
    'GIF/mandelbrot_zoom.gif',
    '-y',

])

for f in glob('GIF/*.png'):
    os.remove(f)