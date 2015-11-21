import pylab
import numpy
import math

def Rgb2Gray(rgb):
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.144]);

def BinaryOtsu(img):
    imgGray = numpy.array(img);
    if (img.ndim  > 2):
        imgGray = Rgb2Gray(img);
    hist, bins = numpy.histogram(imgGray);
    minBrightness = numpy.min(imgGray);
    maxBrightness = numpy.max(imgGray);
    N = imgGray.size;
    Wi = (hist.cumsum(0));
    Mi = (hist*numpy.arange(hist.size)).cumsum(0)/Wi;
    Di = Wi*(N-Wi)*((2*Mi-1/N)**2);
    bound = numpy.argmax(Di);
    boundBrightness = bins[bound];
    imgGray[imgGray < boundBrightness] = 0;
    imgGray[imgGray >= boundBrightness] = 255;
    return numpy.uint8(imgGray);
sample = pylab.imread('sample4.jpg');
m = BinaryOtsu(sample);
pylab.subplot(1, 2, 1),  pylab.imshow(sample, cmap = pylab.get_cmap('gray'));
pylab.subplot(1, 2, 2),  pylab.imshow(m, cmap = pylab.get_cmap('gray'));
pylab.show();
