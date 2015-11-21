import pylab
import numpy
import math

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.144]);

# Функция AutoCorrection
# img исходное изображение
# blackPercents процент темных пикселей
# whitePercents проента белых пикселей
# return imgGray серое изображение с примененной автокоррекцией
def AutoCorrection(img, blackPercents, whitePercents):
    imgGray = rgb2gray(img);
    coutnElements = (imgGray.size) / 100.;
    countWhites = int(coutnElements * whitePercents)-1;
    countBlacks = int(coutnElements * blackPercents)-1;
    linI = numpy.sort(numpy.array(imgGray.reshape(imgGray.size)));
    black = linI[countBlacks];
    white = linI[imgGray.size - countWhites];
    imgGray[imgGray < black] = 0;
    imgGray[imgGray > white] = 255;
    tmp = (imgGray > black) & (imgGray < white);
    d = white - black;
    imgGray[tmp] = 0 + (255 - 0) * (imgGray[tmp]-black) / d;
    return imgGray;

sample = pylab.imread('sample2.jpg');
m = AutoCorrection(sample, 10, 10);
pylab.subplot(1, 2, 1),  pylab.imshow(sample);
pylab.subplot(1, 2, 2),  pylab.imshow(m, cmap = pylab.get_cmap('gray'));
pylab.show();
