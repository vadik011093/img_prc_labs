
# coding: utf-8

# ## imports

# In[98]:

import pylab as pl
import numpy as np
import math
import cv2
from numpy import matlib
from numpy.linalg import inv


# ## functions implementation

# In[99]:

#draw lines on img
#img - image
#lines - array of coordinates
def PaintLines(img, lines):
    for line in lines:
        x1 = (int)(line[1]);
        y1 = (int)(line[0]);
        x2 = (int)(line[3]);
        y2 = (int)(line[2]);
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1);
        
#get hough space from img
#img - image
#return h - hough space, alphas - array of angles, ps - array of radiuses
def Hough(img, isGray = False):
    imgGray = np.array(img);
    if (imgGray.ndim > 2):
        imgGray = cv2.cvtColor(imgGray,cv2.COLOR_BGR2GRAY);
    maxP =(int)(math.sqrt(imgGray.shape[0]**2+imgGray.shape[1]**2));
    maxAlpha = 180;
    dAlpha = 1;
    alphas = np.deg2rad(np.arange(0, maxAlpha, dAlpha));
    cosA = np.cos(alphas);
    sinA = np.sin(alphas);
    nP = 2*maxP+1;
    if (not (isGray)):
        h = np.zeros((alphas.shape[0], nP));
    else:
        h = np.zeros((alphas.shape[0], nP));
    yI, xI = np.nonzero(imgGray);
    for idx in range(yI.shape[0]):
        x = xI[idx];
        y = yI[idx];
        for alphaI in range(alphas.shape[0]):
            p = maxP+(int)(x*cosA[alphaI] + y*sinA[alphaI]);
            if (not (isGray)):
                h[alphaI,p]+=1;
            else:
                h[alphaI,p]+=imgGray[y,x];
                #h[alphaI,p,0]+=1;
    ps = np.arange(-maxP,maxP,1);
    #if (isGray):
    #    h = h[:,:,1]/(h[:,:,0]+1);
    return (h, alphas, ps);

#get hough's maxes for n
#h - hough space
#count - count of maxes
#eps - size of side
#left - count of hough's curves in max or lowest intensity
#return vals - curves in max, alphas - array of max's angles, ps - array of max's radiuses
def GetHoughMaxes(h, count=1, eps=0, left = 0):
    hC = np.array(h);
    p = [];
    alphas = [];
    vals = [];
    cCount = 0;
    if (eps == 0):
        eps = (int)(h.shape[0]/50);
    while (cCount < count):
        yI, xI = np.nonzero(hC == hC.max());
        if ((hC[yI, xI] >= left).any()):
            x = xI[0];
            y = yI[0];
            p.append(x);
            alphas.append(y);
            vals.append(hC[y,x]);
            l = x-eps if x-eps>0 else 0;
            t = y-eps if x+eps<x-eps>0 else 0;
            r = x+eps if x+eps<hC.shape[1]-1 else hC.shape[1];
            b = y+eps if x+eps<hC.shape[0]-1 else hC.shape[0];
            hC[t:b,l:r] = 0.0;
            cCount+=1;
        else:
            cCount = count;
    return (np.array(vals), np.array(alphas), np.array(p));

#get pixels in line
#yI - y-indexes of nonzero in img
#xI - x-indexes of nonzero in img
#p - array of radiuses
#return yI, xI - indexes in line
def GetHoughLinePixel(yI, xI, alphaI, pI, alphas, p):
    alpha = alphas[alphaI];
    pp = p[pI];
    pps = xI*np.cos(alpha)+yI*np.sin(alpha);
    accuracy = 1;
    ps = np.nonzero(np.abs(pps-pp) < accuracy);
    return yI[ps], xI[ps];

#get pixels in lines
#img - image
#alphasIndexes - indexes of angle
#pIndexes - indexes of radius
#alphas - array of angles
#p - array of radiuses
#return yI, xI - indexes in line
def GetHoughLinePixels(img, alphasIndexes, pIndexes, alphas, p):
    lineIndexesY = [];
    lineIndexesX = [];
    yI, xI = np.nonzero(img);
    for idx in range(alphasIndexes.shape[0]):
        alphaI = alphasIndexes[idx];
        pI = pIndexes[idx];
        yIs, xIs = GetHoughLinePixel(yI, xI, alphaI, pI, alphas, p)
        lineIndexesY.append(yIs);
        lineIndexesX.append(xIs);
    return (np.array(lineIndexesY), np.array(lineIndexesX));

#get lines
#lineIndexesY - y-indexes of lines
#lineIndexesX - x-indexes of lines
#alphasIndexes - indexes of angle
#pIndexes - indexes of radius
#alphas - array of angles
#ps - array of radiuses
#return lines - array of starts and ends of lines
def GetHoughLines(lineIndexesY, lineIndexesX, alphasIndexes, pIndexes, alphas, ps, minLength=20):
    lines = [];
    for idx in range(lineIndexesY.shape[0]):
        alpha = math.pi/2-alphas[alphasIndexes[idx]];
        rotate = np.array(((math.cos(alpha), math.sin(alpha)),(-math.sin(alpha),math.cos(alpha))));
        rotateInv = inv(rotate);
        lY = lineIndexesY[idx];
        lX = lineIndexesX[idx];
        m = np.transpose(np.array((lX,lY)));
        rotated = np.dot(m,rotate);
        x0 = rotated[:,0].min();
        x1 = rotated[:,0].max();
        l = x1-x0;
        p = ps[pIndexes[idx]];
        if (l >= minLength):
            p1 = np.dot(np.array((x0,p)),rotateInv);
            p2 = np.dot(np.array((x1,p)),rotateInv);
            lines.append([p1[1],p1[0], p2[1], p2[0]]);
    return lines;


# ## read image and preprocessing

# In[100]:

sample = cv2.imread('img/sample3.jpg');

blur = cv2.GaussianBlur(sample,(5,5),1);

gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY);

edges = cv2.Canny(gray, 50, 150);


# ## get hough space 

# In[101]:

h, alphas, p = Hough(edges);


# ## get hough maxes

# In[102]:

vals, alphasIndexes, pIndexes = GetHoughMaxes(h, 7);


# ## get hough lines pixels

# In[103]:

lineIndexesY, lineIndexesX = GetHoughLinePixels(edges, alphasIndexes, pIndexes, alphas, p);


# ## get lines

# In[104]:

lines = GetHoughLines(lineIndexesY, lineIndexesX, alphasIndexes, pIndexes, alphas, p);


# ## show image with lines

# In[105]:

img = np.array(sample);
PaintLines(img,lines);
pl.subplot(1, 2, 1),  pl.imshow(img, 'gray');


# ## run for gray image

# In[106]:

grayLines = edges*gray;
h, alphas, p = Hough(grayLines, True);


# In[107]:

vals, alphasIndexes, pIndexes = GetHoughMaxes(h, 5);

lineIndexesY, lineIndexesX = GetHoughLinePixels(gray, alphasIndexes, pIndexes, alphas, p);


# In[108]:

lines = GetHoughLines(lineIndexesY, lineIndexesX, alphasIndexes, pIndexes, alphas, p);


# In[109]:

img = np.array(sample);
PaintLines(img,lines);
pl.subplot(1, 2, 2),  pl.imshow(img, 'gray');


# In[110]:

pl.show();



