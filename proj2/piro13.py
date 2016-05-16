import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import math
import copy
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon
import random
       
def rotate_img(img, angle):
    return ndimage.rotate(img, angle)

def ourRound(number):
    return round(float(number)/200)

def printResults(results):
    for i in range(0,len(results)):
        for j in range(0,len(results[0])):
            sys.stdout.write(str(int(results[i][j])))
            sys.stdout.write(' ')
        sys.stdout.write('\n')
        
def loadFullImage(path,L,K):
    filename=path+'image.png'
    img = cv2.imread(filename,0)
    rows=L*200
    cols=K*200
    right=img.shape[1]
    bottom=img.shape[0]
    pts1 = np.float32([[right,0],[right,bottom],[0,bottom],[0,0]])
    pts2 = np.float32([[cols,0],[cols,rows],[0,rows],[0,0]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(cols,rows))
    return dst
    
def transformCorners(img,rows,cols, rgb):
    pts1 = np.float32(findCorners(rgb))
    pts2 = np.float32([[0,rows],[0,0],[cols,0],[cols,rows]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(cols,rows))
    return dst
    
def loadImages(L,K,path,imgFull):
    images = []
    rows=200
    cols=200
    for i in range(0,(L*K)):
        filename=path+str(i)+'.png'
        img = cv2.imread(filename, 0)
        rgb = cv2.imread(filename, -1)
        img=transformCorners(img,rows,cols, rgb)
        images.append(img)
    return images

def findExtremes(img):    
    min_y = 10000
    max_y = -1
    min_x = 10000
    max_x = -1
    topmost=[0,min_y]
    rightmost=[max_x,0]
    bottommost=[0,max_y]
    leftmost=[min_x,0]
    for y in range(0, img.shape[0]):
       for x in range(0, img.shape[1]):
          if img[y,x] > 0:
             if y < min_y:
                min_y = y
                topmost=[x,y]
             if y > max_y:
                max_y = y
                bottommost=[x,y]
             if x > max_x:
                max_x = x
                rightmost=[x,y]
             if x < min_x:
                min_x = x
                leftmost=[x,y]
    return topmost,rightmost,bottommost,leftmost

def odl(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def sumOdl(tab, a):
    sum = 0
    for t in tab:
        sum += odl(a, t)
    return sum

def createLine(tab, a, b, c):
    if ( odl(a, b) + odl(b, c) ) * 0.95 > odl(a, c):
        return False
    else:
        if sumOdl(tab, a) > sumOdl(tab, b) and sumOdl(tab, c) > sumOdl(tab, b):
            return True
        else:
            return False

def aprox(tab):
    tab = tab[:-1]
    isIn = np.ones((len(tab)))
    for i in range(0, len(tab)):
        if createLine(tab, tab[i-2], tab[i-1], tab[i]):
            isIn[i-1] = 0  
    return tab[isIn == 1]

def findCorners(img):
    result=[]
    img1 = img[:,:,-1]
    img2 = np.zeros((img1.shape[0] + 20, img1.shape[1] + 20))
    img2[10:img1.shape[0] + 10, 10:img1.shape[1] + 10] = img1
    contours = find_contours(img2, 0)
    aprx = approximate_polygon(contours[0], tolerance=10.0)
    aprx4 = aprox(aprx)
    for i in range(0, 4):
        result.append(aprx4[i].tolist())
    for i in range(0,len(result)):
        result[i].reverse()
    for i in range(0, len(result)):
        for j in range(0, len(result[i])):
            result[i][j] -= 10
    return result

def firstNotUsed(L, K, results):
    isIn = np.ones((L*K))
    for y in range(0, L):
        for x in range(0, K):
            if results[y][x] != -1:
                isIn[results[y][x]] = 0
    for i in range(0, L*K):
        if isIn[i] == 1:
            return i
    return None

def iteration(L,K,imgFull, imgList, results):
    results2 = results.copy()
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    method = eval(methods[3])
    extremis = np.zeros((L,K))
    for i in range(0,len(imgList)):
        template = imgList[i]
        template2 = np.asarray(template)
        extremum = 0
        index = 0
        e_top_left = 0
        extremum = 0
        for j in range(0,4):
            angle=90*j
            template3=rotate_img(template2,angle)
            res = cv2.matchTemplate(imgFull,template3,method)
            res = cover(L, K, res, results2)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            if max_val > extremum:
                extremum = max_val
                index = j
                e_top_left = top_left
        yy = ourRound(e_top_left[1])
        xx = ourRound(e_top_left[0])
        if results[yy][xx] == -1 and extremis[yy][xx] < extremum:
            results[yy][xx] = firstNotUsed(L, K, results)
            extremis[yy][xx] = extremum
        imgList[i] = rotate_img(imgList[i], index * 90)
    return results

def blind(img, L, K, y, x):
    for y1 in range(200):
        for x1 in range(200):
            img[min((L-1)*200, max(0, y*200 + y1 - 100))][min((K-1)*200, max(0, x*200 + x1 - 100))] = 0
    return img

def cover(L, K, img, res):
    for y in range(0, L):
        for x in range(0, K):
            if res[y][x] != -1:
                img = blind(img,L, K, y, x)
    return img

def getResults(L,K,imgFull,imgList):
    orginalImgList = copy.deepcopy(imgList)
    results = np.empty((L,K))
    results[:] = -1
    while True:
        results = iteration(L,K,imgFull,imgList, results)
        isIn = np.ones((L*K))
        for y in range(0, L):
            for x in range(0, K):
                if results[y][x] != -1:
                    isIn[results[y][x]] = 0
        imgList2 = []
        for i in range(0, L*K):
            if isIn[i] == 1:
                imgList2.append(orginalImgList[i])
        imgList = imgList2
        if len(imgList) == 0:
            break
    return results 

def main():
    path=sys.argv[1]
    L=int(sys.argv[2])
    K=int(sys.argv[3])
    imgFull=loadFullImage(path,L,K)
    imgList=loadImages(L,K,path,imgFull)
    results=getResults(L,K,imgFull,imgList)
    printResults(results)
        
main()
