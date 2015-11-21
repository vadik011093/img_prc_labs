import numpy
import pylab
import math

import cv2
def BoxFilter(img, d):
    s = d * d;
    imgCopy = np.array(img)/255;
    res = np.array(imgCopy);
    [height, width, colors] = imgCopy.shape;
    if ((d > 1) and (d < height) and (d < width)):
        imgCopy = cv2.integral(imgCopy)[1:,1:,:];
        d = (int)(d / 2);
        for y in range(height):
            for x in range(width):
                left = x - d;
                right = x + d;
                top = y - d;
                bottom = y + d;
                if (left < d):
                    left = 0;
                q
                if (right + 1 > width):
                    right = width - 1;
                    
                if (top < 0):
                    top = 0;
                
                if (bottom + 1 > height):
                    bottom = height - 1;
                
                s = (right - left) * (bottom - top);
                
                res[y, x] = (imgCopy[top, left] + imgCopy[bottom, right] - imgCopy[bottom, left] - imgCopy[top, right]) / s;        
    return res;

sample = pylab.imread('img/sample5.jpg');
m = BoxFilter(sample,4);
pylab.subplot(1, 3, 1),  pylab.imshow(sample);
pylab.subplot(1, 3, 2),  pylab.imshow(m);
pylab.show();