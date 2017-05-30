# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:46:39 2017
Minimum surface needed to register successfully
@author: sghanavati
All points are 3xM
"""

from optparse import OptionParser
import numpy as np
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
from scipy import optimize    ###curve_fit, leastsq

from icp_functions import *

from sklearn import mixture
import random

def gmmfunc(gmix, x, y, z):
    gmm = 0
    for i in range(gmix.weights_.size):
        gmm += gmix.weights_[i]*guassian3d(gmix.means_[i], np.matrix(gmix.covars_[i]),np.array([x, y, z]))
    return gmm
    
def guassian3d(mu, covarmat, p):
    a = np.matrix(p-mu).T   #matrix 3x1
    return 1.0/np.sqrt(np.power(2*np.pi,3)*np.linalg.det(covarmat))*np.exp(-0.5*a.T*np.linalg.inv(covarmat)*a)
    
#def GaussTransform(X, fixedPnts, movingPnts):    #trx, try, trz, ax, ay, az, tuple((fixedPnts, movingPnts, unit))
#    movingPnts = rigidtransform(movingPnts, X[0], X[1], X[2], X[3], X[4], X[5], 'deg')
#    fgmix = mixture.GMM(n_components=3, covariance_type='full')
#    fgmix.fit(fixedPnts)      
#    mgmix = mixture.GMM(n_components=3, covariance_type='full')
#    mgmix.fit(movingPnts) 
#    costfunc = 0
#    for i in range(fgmix.weights_.size):
#        for j in range(mgmix.weights_.size):
#            costfunc -= fgmix.weights_[i]*mgmix.weights_[j] * np.exp(-np.linalg.norm(fgmix.means_[i]-movingPnts[j])/(np.linalg.det(fgmix.covars_[i]) + np.linalg.det(mgmix.covars_[j])))
#    return costfunc
     
#def GaussTransform(X, fixedPnts, movingPnts):    #trx, try, trz, ax, ay, az, tuple((fixedPnts, movingPnts, unit))
#    movingPnts = rigidtransform(movingPnts, X[0], X[1], X[2], X[3], X[4], X[5], 'deg')
#    costfunc = 0
#    gradientcost = 0
#    for a in fixedPnts:
#        for b in movingPnts:
#            costfunc += np.exp(-np.linalg.norm(a-b))
#            gradientcost -= 2.0 * np.exp(-np.linalg.norm(a-b)) * (a-b)
#    return costfunc

def GaussTransform2(X, fgmix, mgmix):    #trx, try, trz, ax, ay, az, tuple((fixedPnts, movingPnts, unit))
    movingPnts = rigidtransform(mgmix.means_, X[0], X[1], X[2], X[3], X[4], X[5], 'deg')
    #print("GaussTransform2 ", movingPnts)
    costfunc = 0
    for i in range(fgmix.weights_.size):
        for j in range(mgmix.weights_.size):
            costfunc -= fgmix.weights_[i]*mgmix.weights_[j] * np.exp(-np.linalg.norm(fgmix.means_[i]-movingPnts[j])/(np.linalg.det(fgmix.covars_[i]) + np.linalg.det(mgmix.covars_[j])))
            #print("costfunc ", costfunc)
    return costfunc
        
if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [options] inputpoints.obj ",
                          version="%prog 1.0")
    parser.add_option("-s", "--skip",
                      type='int',
                      action="store", 
                      dest="skip", 
                      default=1, 
                      help="The number of points to skip in between every 2 points of the input pointset.")
    parser.add_option("-n", "--noise",
                      action="store_true", 
                      dest="noise", 
                      default=False, 
                      help="Add noise to data")
    parser.add_option("-v", "--vertebrae",
                      type='choice',
                      action="store", 
                      dest="vertebrae", 
                      default='s', 
                      choices=['s','p','g'],
                      help="The choice for multi_vertebrae registration. s:single body, p:piecewise, g:groupwise(with biomechanical constraints)")
    parser.add_option("-r", "--repeat",
                      type='int',
                      action="store", 
                      dest="repeat_icp", 
                      default=0, 
                      help="The number of times you want to repeat icp.")
    
    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.error("wrong number of arguments")

    print(options)
    print(args)
    
    fixedPnts = read1obj (args[0], options.skip)   #array Nx3            
    movingPnts = read1obj (args[0], options.skip)  #array Mx3
    gttr = np.array([random.uniform(-10,10), random.uniform(-10,10), random.uniform(-10,10)])
    gtrot = np.array([random.uniform(-20,20), random.uniform(-20,20), random.uniform(-20,20)])
    gtR = rotationX(gtrot[0], unit='deg')*rotationY(gtrot[1], unit='deg')*rotationZ(gtrot[2], unit='deg')
    #movingPnts = np.array((gtR*movingPnts.T)).T   #array Nx3  
    #movingPnts += np.tile(gttr, (movingPnts.shape[0], 1))
    print("ground truth translation ", gttr)
    print("ground truth rotation ", gtrot)
    print("ground truth rotation matrix ", gtR)
    
    movingPnts = rigidtransform(movingPnts, gttr[0], gttr[1], gttr[2], gtrot[0], gtrot[1], gtrot[2], unit='deg')
    
    #add noise
    fixedPnts = addnoise(fixedPnts)
    movingPnts = removepoints(movingPnts, 0.05)
    movingPnts = addnoise(movingPnts, 0.05)
    
    
    view3d1(fixedPnts, movingPnts, 'Before registration')
    
    #### icp
    TRicp, Tticp, itert, itererr, icptime, icperr, iternum, f, m, findices  = icp(fixedPnts, movingPnts)
    registered_movingPnts = np.array(m)
    print("\nthe ICP translation: \n", Tticp)     #array 3x1
    print("\nthe ICP rotation: \n", TRicp)        #array 3x3
    print("\ninitial rmse = ", itererr[0], "\n")
    print("\nthe ICP rmse: ", icperr)       
    print("\n",iternum, " iterations of the ICP took ", icptime,"s\n") 
    writeobj (args[0][:-4]+"_icpreg.obj",registered_movingPnts)
    view3d1(fixedPnts, registered_movingPnts , 'After ICP')

    
    #gmm
    #fixed
    fgmix = mixture.GMM(n_components=3, covariance_type='full')
    fgmix.fit(fixedPnts)      
    print(fgmix.means_)
    print(fgmix.covars_)
    print(fgmix.weights_)
    fcolors = ['r' if i==0 else 'g' if i==1 else 'c' for i in fgmix.predict(fixedPnts)]
    
    #moving
    gmix = mixture.GMM(n_components=3, covariance_type='full')
    gmix.fit(movingPnts)      
    print(gmix.means_)
    print(gmix.covars_)
    print(gmix.weights_)
    colors = ['r' if i==0 else 'g' if i==1 else 'c' for i in gmix.predict(movingPnts)]
    #colors = ['r', 'g', 'b']
    fig = plt.figure()
    fig.suptitle("GMM", fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0,fixedPnts.shape[0],100):
        ax.scatter(fixedPnts[i][0], fixedPnts[i][1], fixedPnts[i][2], c=fcolors[i], marker='x')
    for i in range(0,movingPnts.shape[0],100):
        ax.scatter(movingPnts[i][0], movingPnts[i][1], movingPnts[i][2], c=colors[i], marker='o')
        
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()        

    X = np.arange(-5, 5, 1)
    Y = np.arange(-5, 5, 1)
    Z = np.arange(-5, 5, 1)
    #X, Y, Z = np.meshgrid(X, Y, Z)
    
    fgmm_centroid = 0
    mgmm_centroid = 0
    for i in range(fgmix.weights_.size):
        fgmm_centroid += fgmix.weights_[i]*fgmix.means_[i]
        mgmm_centroid += gmix.weights_[i]*gmix.means_[i]

    inittr = fgmm_centroid - mgmm_centroid
    print("gmm init translation ", inittr)
    initrot = np.array([0.0, 0.0, 0.0])
    
    centroid_f = np.mean(fixedPnts, axis=0)
    centroid_m = np.mean(movingPnts, axis=0)        
    inittranslation = centroid_f-centroid_m
    print("centroid init translation ", inittranslation)
    

    #xopt, fpot, numiter  = optimize.fmin(GaussTransform, np.array([inittranslation[0], inittranslation[1], inittranslation[2], 0.0, 0.0, 0.0]), args = (fixedPnts, movingPnts), maxiter=10)
    xopt  = optimize.fmin(GaussTransform2, np.array([inittranslation[0], inittranslation[1], inittranslation[2], 0.0, 0.0, 0.0]), args = (fgmix, gmix), maxiter=1000)

    movingPnts = rigidtransform(movingPnts, xopt[0], xopt[1], xopt[2], xopt[3], xopt[4], xopt[5], unit='deg')
    view3d1(fixedPnts, movingPnts , 'After GMM REG')
    writeobj (args[0][:-4]+"_gmmreg.obj",movingPnts)
    
    print(xopt)
    #print(fopt)
    #print(numiter)
    print("ground truth translation ", gttr)
    print("ground truth rotation ", gtrot)
    print("ground truth rotation matrix ", gtR)

    print("\nthe ICP translation: \n", Tticp)     #array 3x1
    print("\nthe ICP rotation: \n", TRicp)        #array 3x3
    
    '''
    costfunc = 0
    gradientcost = 0
    for a in fixedPnts:
        for b in movingPnts:
            costfunc += np.exp(-np.norm(a-b))
            gradientcost -= 2.0 * np.exp(-np.norm(a-b)) * (a-b)
    
    
    for x in X:
        for y in Y:
            for z in Z:
                costfunc -= gmmfunc(gmix, x, y, z)*gmmfunc(fgmix, x, y, z)

    print("costfunc ", costfunc)
    '''