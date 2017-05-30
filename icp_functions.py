# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:58:09 2016

@author: sghanavati

We need to segment the vertebrae from the pre-op CT/MR scan
We will need to removal outlier points from the surfaces using RANSAC
We will need to separate the vertebrae from each other
"""
#!/usr/bin/env python

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



#1. read the obj files vertices
def read1obj (fname, skipl = 1):
    digits = ['0', '1', '2', '3', '4', '5', '6', '7' , '8', '9']
    vertebra_vs = []
    cnt = 0
    with open(fname, "r") as f:
        for line in f:
            if (cnt % skipl)==0:
                content = line.split(' ')
                if content[0]=="v":
                    #vertebra_vs.append([float(content[1]),float(content[2]),float(content[3])])
                    thisv = [float(i) for i in content if True in [d in i for d in digits]]
                    vertebra_vs.append(thisv)
            cnt = cnt + 1    
        return np.array(vertebra_vs)
    
    
def readobj (fname, skipl = 1, body = 's'):
    digits = ['0', '1', '2', '3', '4', '5', '6', '7' , '8', '9']
    multivertebra_vs = []
    vertebra_vs = []
    cnt = 0
    with open(fname, "r") as f:
        for line in f:
            if body=='s':
                if (cnt % skipl)==0:
                    content = line.split(' ')
                    if content[0]=="v":
                        #vertebra_vs.append([float(content[1]),float(content[2]),float(content[3])])
                        thisv = [float(i) for i in content if True in [d in i for d in digits]]
                        vertebra_vs.append(thisv)
                cnt = cnt + 1
            else:
                content = line.split(' ')
                if content[0][0]=="#":
                    cnt = 0
                    if (len(vertebra_vs)>0):
                        multivertebra_vs.append(vertebra_vs)
                        vertebra_vs = []
                elif content[0]=="v":
                    cnt = cnt + 1
    
    if body=='s':
        return np.array(vertebra_vs)
    if (len(vertebra_vs)>0):
        multivertebra_vs.append(np.array(vertebra_vs))
                
    multivertebra_vs = np.array(multivertebra_vs)
    return multivertebra_vs
    
#write 1 vertebra at a time    
def writeobj (fname,vertebra_vs):
    with open(fname, "w+") as f:
        f.write("#created by python code written by Sahar\n")
        for i in range(vertebra_vs.shape[0]):
            f.write("v "+str(vertebra_vs[i][0])+" "+str(vertebra_vs[i][1])+" "+str(vertebra_vs[i][2])+"\n")               
    return 0

#add noise to 1 vertebra at a time    
def addnoise (vertebra_vs, nportion = 0.01):    #array Nx3
    noisenum = int(nportion*vertebra_vs.shape[0])
    noisyvertices = list(vertebra_vs)
    for i in range(noisenum):
        #indx = np.random.randint(0,vertices.shape[0])
        #noisyvertices.append([vertices[indx][0]-np.random.uniform(np.amin(vertices),vertices[indx][1]-np.amax(vertices)) , vertices[indx][2]-np.random.uniform(np.amin(vertices),np.amax(vertices)) , np.random.uniform(np.amin(vertices),np.amax(vertices))]])
        #noisyvertices.append([np.random.uniform(np.amin(vertices),np.amax(vertices)) , np.random.uniform(np.amin(vertices),np.amax(vertices)) , np.random.uniform(np.amin(vertices),np.amax(vertices))]])
        noisyvertices.append([np.random.uniform(np.amin(vertebra_vs),np.amax(vertebra_vs)) , np.random.uniform(np.amin(vertebra_vs),np.amax(vertebra_vs)) , np.random.uniform(np.amin(vertebra_vs),np.amax(vertebra_vs))])
    noisyvertices = np.array(noisyvertices)
    shuffle( noisyvertices )
    return noisyvertices


#add noise to 1 vertebra at a time    
def removepoints (vertebra_vs, nportion = 0.01):       #array Nx3 
    N = vertebra_vs.shape[0]
    noisenum = int(nportion*N)
    for i in range(noisenum):
        indx = np.random.randint(0,N)
        N = N-1
        np.delete(vertebra_vs, indx,0)  #remove a row
    shuffle( vertebra_vs )
    return vertebra_vs

    
#2. show 3D points
def view3d1(pointset1, pointset2,plttitle):
    print("view3d ", plttitle, pointset1.shape[0], pointset2.shape[0])
    fig = plt.figure()
    fig.suptitle(plttitle, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0,pointset1.shape[0],100):
        ax.scatter(pointset1[i][0], pointset1[i][1], pointset1[i][2], c='b', marker='x')
    for i in range(0,pointset2.shape[0],100):    
        ax.scatter(pointset2[i][0], pointset2[i][1], pointset2[i][2], c='r', marker='o') #marker='^'
        
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    return 0

def view3d(pointsets1, pointsets2,plttitle):
    print("view3d ", plttitle, pointsets1.shape[0], pointsets2.shape[0])
    fig = plt.figure()
    fig.suptitle(plttitle, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111, projection='3d')
    for p1 in pointsets1:
        pointset1 = np.array(p1)
        for i in range(0,pointset1.shape[0],100):
            ax.scatter(pointset1[i][0], pointset1[i][1], pointset1[i][2], c='b', marker='x')
    for p2 in pointsets2:
        pointset2 = np.array(p2)
        for i in range(0,pointset2.shape[0],100):    
            ax.scatter(pointset2[i][0], pointset2[i][1], pointset2[i][2], c='r', marker='o') #marker='^'
        
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    return 0

def quaternion_matrix(q):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    #q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < 1e-10:
        return np.identity(4)
    q *= sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])
    
    
def rotationX(t, unit='rad'):
    if unit=='deg':
        t = t*np.pi/180.
    return np.matrix([[1, 0, 0],
                      [0, np.cos(t), -np.sin(t)],
                      [0, np.sin(t), np.cos(t)]])

def rotationY(t, unit='rad'):
    if unit=='deg':
        t = t*np.pi/180.
    return np.matrix([[np.cos(t), 0, np.sin(t)],
                      [0, 1, 0],
                      [-np.sin(t), 0, np.cos(t)]])

def rotationZ(t, unit='rad'):
    if unit=='deg':
        t = t*np.pi/180.
    return np.matrix([[np.cos(t), -np.sin(t), 0],
                      [np.sin(t), np.cos(t), 0],
                      [0, 0, 1]])
    
def rigidtransform(movingPnts, tx, ty, tz, ax, ay, az, unit='rad'):  #array Mx3
    rotation = rotationX(ax, unit) * rotationY(ay, unit) * rotationZ(az, unit)  
    translation = np.array([tx, ty, tz])
    movingPnts = np.array((rotation*movingPnts.T)).T   #array Nx3  
    movingPnts += np.tile(translation, (movingPnts.shape[0], 1))
    return movingPnts

    
def rotationbyvectorangle(u, t, unit='rad'):
    #reference: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    u = u/np.linalg.norm(u)
    if unit=='deg':
        t = t*np.pi/180.
    return np.matrix([[np.cos(t)+u[0]*u[0]*(1-np.cos(t)), u[0]*u[1]*(1-np.cos(t))-u[2]*np.sin(t), u[0]*u[2]*(1-np.cos(t))+u[1]*np.sin(t)],
                      [u[0]*u[1]*(1-np.cos(t))+u[2]*np.sin(t), np.cos(t)+u[1]*u[1]*(1-np.cos(t)), u[1]*u[2]*(1-np.cos(t))+u[0]*np.sin(t)],
                      [u[0]*u[2]*(1-np.cos(t))+u[1]*np.sin(t), u[1]*u[2]*(1-np.cos(t))+u[0]*np.sin(t), np.cos(t)+u[2]*u[2]*(1-np.cos(t))]])
    
def shuffle( X ):
    N = X.shape[0]  #array Nx3
    idx = [i for i in range(N)]     #range( N )
    np.random.shuffle( idx )
    return X[idx]
     
def select_n_from_N2darray_norepeat(inarray2d, n):
    #no repeat: while (0 in np.sum(tempf - f[indx],axis=1)):       # f[indx] in tempf
    ptemp = [list(x) for x in list(inarray2d)]
    pselected = []
    for i in range(n):
        indx = np.random.randint(0,len(ptemp))
        pselected.append(ptemp[indx])
        ptemp.remove(ptemp[indx])
    return np.array(pselected)

    
def select_n_from_N2darray_repeat(inarray2d, n):
    pselected = []
    for i in range(n):
        indx = np.random.randint(0,inarray2d.shape[0])
        pselected.append(inarray2d[indx])
    return np.array(pselected)

 
def rmse_calc(f,m):
    # Find the error
    err = m - f  #[x-x' y-y' z-z'] array Nx3    
    err = np.multiply(err, err)     # [(x-x')^2 (y-y')^2 (z-z')^2] array Nx3
    err = sum(sum(err))                # sum(array 1x3)
    rmse = sqrt(err/m.shape[0])
    return rmse

def partial_rmse_calc(f,m):
    err = m - f  #[x-x' y-y' z-z'] array Nx3    
    err = np.multiply(err, err)     # [(x-x')^2 (y-y')^2 (z-z')^2] array Nx3
    err = sum(sum(err))                # sum(array 1x3)
    #rmse = sqrt(err/m.shape[0])
    return err, m.shape[0]
    
'''    
# Input: expects Nx3 matrix of points
# m:moving points
# f:fixed points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector
# R*m+t = f
# [U,S,V] = svd((m-mcentroid) * (f-fcentroid).T)
# R = V*U.T
# t = fcentroid - R*mcentroid
'''
def svd_rigid_transform_3D(f,m): #(A, B):
    #assert f.shape[0] == m.shape[0]
    N = min(f.shape[0], m.shape[0]) # total points

    #if the len of the points are not the same, randomly pick N points from the larger pointset
    if (f.shape[0] > m.shape[0]):
        f = select_n_from_N2darray_norepeat(f, m.shape[0])
        
    elif (m.shape[0] > f.shape[0]):
        m = select_n_from_N2darray_norepeat(m, f.shape[0])
        
    centroid_f = np.mean(f, axis=0)
    centroid_m = np.mean(m, axis=0)
    
    # centre the points
    ff = f - np.tile(centroid_f, (N, 1))
    mm = m - np.tile(centroid_m, (N, 1))
    
    ###let's check the eigenvectors
    hff = np.matrix(np.transpose(ff)) * np.matrix(ff) 
    U, S, Vt = np.linalg.svd(hff)
    #print("\n\n>>>>>>>>>>>>>>>>>>>>>eigenVectors fixed: ")
    #print(U)
    mff = np.matrix(np.transpose(mm)) * np.matrix(mm) 
    U, S, Vt = np.linalg.svd(mff)
    #print("\n\n>>>>>>>>>>>>>>>>>>>>>eigenVectors moving: ")
    #print(U)


    # dot is matrix multiplication for array
    H = np.matrix(np.transpose(mm)) * np.matrix(ff)       # H is a 3x3 matrix.

    # since H is matrix, so will be U,S,Vt
    U, S, Vt = np.linalg.svd(H)	#The SVD is commonly written as a = U S V.H. The v returned by this numpy.linalg.svd function is V.H. If U is a unitary matrix, it means that it satisfies U.H = inv(U).

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
       print( "Reflection detected")
       Vt[2,:] *= -1
       R = Vt.T * U.T

    tr = np.array(-R*np.matrix(centroid_m).T + np.matrix(centroid_f).T)   # matrix 3x3
    t = np.matrix([tr[0][0],tr[1][0],tr[2][0]]).T   # matrix 3x1
    
    return R, t, f, m   #return fixed and moving points of the same size
    
  
def icp(f,m, maxiter = 100, errthresh = 1e-5):
    print("icp registration")
    #point matching using KDTree
    TR = np.eye(3)      #total Rotation
    Tt = np.zeros((3,1))    #total translation
    tic = time()
    iternum = 0
    itert = []
    itererr = []
    f = np.array(f)  #array Nx3
    m = np.array(m)  #array Mx3
    N = f.shape[0]  #Nx3
    M = m.shape[0]  #Mx3
    print(N, " fixed points ", M, " moving points.")
    '''
    #for each point in m: 
    #1.finds the distance to the nearest point in f and saves in corresponding distance vector, 
    #2.finds the index of the closest point in f and saves in corresponding index vector
    '''
    tree_fixed = cKDTree(f)  #data : (N,K) array_like K=3
    corresdist, corresindx = tree_fixed.query(m)   
    itererr.append(rmse_calc(f[corresindx],m))
    
    for i in range(maxiter):
        #if ( (len(itererr)<=2) or ((i>1) and (len(itererr)>2) and (abs(itererr[i]-itererr[i-1])>errthresh)) ) :   
        if ( (len(itererr)<=10) or (abs(itererr[-1]-itererr[-2])>errthresh) ) :   
            print(i, itererr[-1])
            corresdist, corresindx = tree_fixed.query(m)   
            R, t , tempf, tempm = svd_rigid_transform_3D(f[corresindx], m)
            TR = R*TR       #calculating total Rotation # matrix 3x3
            Tt = t+R*Tt     #calculating total translation # matrix 3x1
            m = (R*m.T) + np.tile(t, (1, m.shape[0]))
            m = np.array(m.T)
            itererr.append(rmse_calc(f[corresindx],m))
            itert.append(time()-tic)
            iternum = iternum+1
        
    icptime = time()-tic
    corresdist, corresindx = tree_fixed.query(m)   
    icperr = rmse_calc(f[corresindx],m)
    return TR, Tt, itert, itererr, icptime, icperr, iternum, f, m, corresindx   
    
#for i-th vertebra: f_i and m_i     
def costfunc(f, m, rot, trans, biomech=False):
    R = rotationX(rot[0], unit='rad') * rotationY(rot[1]) * rotationZ(rot[2])
    m = (R*m.T) + np.tile(t, (1, m.shape[0]))
    m = np.array(m.T)    
    cost = rmse_calc(f,m)
    if (biomech):
        cost = cost + springcost
    return cost
    
#for i-th vertebra: spring between m_i-1, m_i, m_i+1     
def springcost(f, m):
    return 
    
