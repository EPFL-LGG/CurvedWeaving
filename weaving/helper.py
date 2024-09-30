# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 18:55:10 2019

@author: chenti
"""

from math import cos, sin, pi,sqrt,acos, tan, radians, degrees

###########
########### HELPER FUNCTIONS 
###########
    
# Get the mid point of two points
def midPt(p0,p1):
    return [(p0[i]+p1[i])/2. for i in [0,1,2]]
    
# Collect unique points
def uniquePt(p0,p1,p2):
    pt=[]
    
    for p0i,p1i,p2i in zip(p0,p1,p2):
        if p0i not in pt:
            pt.append(p0i)
        if p1i not in pt:
            pt.append(p1i)
        if p2i not in pt:
            pt.append(p2i)
    return pt
    
    
def triCalc(rList, arc):
    
    aList=[]
    aList=[chord(arc,r) for r in rList]
    [a,c,b]=sss([aList[1],aList[2],aList[3]])
    return [a,c,b]

def chord(a,r):
    if r==None:
        return a
    else:
        return 2*r*sin(a/(2*r))
    
# Get internal angles from edge lengths
def sss(l):
    c=acos((l[0]**2+l[1]**2-l[2]**2)/(2*l[0]*l[1]))
    a=acos((l[1]**2+l[2]**2-l[0]**2)/(2*l[1]*l[2]))
    b=acos((l[2]**2+l[0]**2-l[1]**2)/(2*l[2]*l[0]))
    return [a,b,c]
    
def vecRot(pt,c,a): # rotate pt w.r.t. to c with angle a
    return [(pt[0]-c[0])*cos(a)+(pt[1]-c[1])*sin(a)+c[0],-1*(pt[0]-c[0])*sin(a)+(pt[1]-c[1])*cos(a)+c[1],pt[2]]
    
    
def angleDot(v0,v1,v2,v3):
    v01=vecSub(v0,v1)
    v23=vecSub(v2,v3)
    return vecAngle2(v01,v23)
    
def vecAngle2(v0,v1):
    return acos(vecDot(v0,v1)/(vecMag(v0)*vecMag(v1)))

def vecAdd(v0,v1):
    return [v0[0]+v1[0],v0[1]+v1[1],v0[2]+v1[2]]
    
def vecSub(v0,v1):
    return [v1[0]-v0[0],v1[1]-v0[1],v1[2]-v0[2]]
        
def vecMag(v):
    return sqrt(v[0]**2+v[1]**2+v[2]**2)
    
def vecNeg(v):
    return [-1*v[0],-1*v[1],-1*v[2]]
def vecNegTree(vt):
    out=[]
    for v in vt:
        out.append(vecNeg(v))
    return out
def vecDot(v0,v1):
    return v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2]
    
def flattenList(p):
     return [item for sublist in p for item in sublist]
     
def drawPoly(n,l):
    a=[radians(ai*360/n) for ai in range(0,n)]
    pini=[l,0,0]
    pi=[vecRot(pini,[0,0,0],ai) for ai in a]
    pino=[0,2*l*cos(a[1]/2),0]
    po=[vecRot(pino,[0,0,0],ai) for ai in a]
    return [pi,po]
    
def vecUnit(v):
    vMag=vecMag(v)
    return [v[0]/vMag,v[1]/vMag,v[2]/vMag]
    
def ppDist(p0,p1):
    return sqrt((p1[0]-p0[0])**2+(p1[1]-p0[1])**2+(p1[2]-p0[2])**2)
    
def alterSign(s):
    if s==1:
        return -1
    else:
        return 1

def angle2vec(a):
    x=1/(sqrt(tan(a)**2+1))
    y=x*tan(a)
    return vecUnit([x,y,0])

def perpVec(a):
    a1=angle2vec(a)
    return [a1[1],-1*a1[0],a1[2]]
    
def movePt(pt,v,l):
    return [pti+vi*l for pti,vi in zip(pt,v)]
    
def mirrorPt(pt, plane):
    if plane=='xz':
        return [[pti[0],-pti[1],pti[2]] for pti in pt]  
    if plane=='yz':
        return [[-pti[0],pti[1],pti[2]] for pti in pt]  
    if plane=='xy':
        return [[pti[0],pti[1],-pti[2]] for pti in pt]  

def flipList(p):
    return [p[i] for i in range(len(p)-1,-1,-1)]        

def angleSum(T, kList):
    angList=[]
    for i,Ti in enumerate(T):
        if kList[i] is not 0:
            angList.append(vecAngle2(Ti[0],vecNeg(Ti[2])))
        else:
            angList.append(0)
    return list(np.cumsum(angList))

def constraintGen(kList, P, n):
    P=list(map(list, zip(*P)))
    [p0,p1,p2]=P
    
    # Find straight line distance between end points of each arc
    tri = [ppDist(p0i,p2i) for p0i,p2i in zip(p0,p2)]
    
    # Find mean distance
    meanLen = sum(tri)/len(tri)
    
    # Find coordinates of the polygon
    [pIn,pOut] = drawPoly(6,meanLen)
    
    # Translate the arcs such that its centroid coincides with one edge of the polygon
    p0t=[]
    p1t=[]
    p2t=[]
    if len(p1)%2==0:
        mid=int((len(p1)+1)/2)
    else:
        mid=int((len(p1))/2)    
    
    midPt0=midPt(pOut[0],pOut[2]) # Mid point of an edge of the polygon
    midPt1=midPt(p0[0],p2[-1]) # Mid point (x) of the three arcs
    midPt1=midPt(midPt1, p1[mid]) # Mid point (y) of the three arcs
    transVec = vecSub(midPt1, midPt0) # Get a translation vector
    
    for p0i,p1i,p2i in zip(p0,p1,p2): # Move the points individually
        p0t.append(movePt(p0i,transVec,1))
        p1t.append(movePt(p1i,transVec,1))
        p2t.append(movePt(p2i,transVec,1))
    
    gRot = angleDot(p0[0],p2[-1],pOut[0],pOut[2]) # Rotation angle between polygon edge and the arcs
    
    p0tr=[]
    p1tr=[]
    p2tr=[]
    
    # Rotate each arc to be parallel to the polygon edge
    for p0i,p1i,p2i in zip(p0t,p1t,p2t):
        p0tr.append(vecRot(p0i,midPt0,gRot))
        p1tr.append(vecRot(p1i,midPt0,gRot))
        p2tr.append(vecRot(p2i,midPt0,gRot))
     