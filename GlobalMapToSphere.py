# author  - Crab
# License - LGPLv3 - www.gnu.org/licenses/lgpl-3.0.html
# github https://github.com/CrabTerlDc/MapToSphere.git/trunk/GlobalMapToSphere.py

import os
import glob
from multiprocessing import Pool, cpu_count

Processes=cpu_count()
#Processes=1 # for dev it's easier
if (Processes <= 1):
    Processes = 1

try:
    # https://pillow.readthedocs.io/en/4.0.x/reference/Image.html
    from PIL import Image, ImageDraw
except:
    print ("consider \"pip install Pillow\"")

try:
    # OpenSource Computer Vision
    import cv2
    #from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
except:
    print ("consider \"pip install opencv-python\"")
    # rem : got "WARNING: The script f2py.exe is installed in 'c:\python38-32\Scripts' which is not on PATH."

try:
    from bigfloat import *
    Pr=BigFloat.exact('0.0', precision=100)
    Epsilon = BigFloat.exact('0.000000000000000000001', precision=100)
except:
    Pr =0.0
    Epsilon = 0.000000001

import math
Pi = math.pi

import time

start_time = time.time()
pict_time = 0


""" rayon de la sphere en m """
ThicMaterial=Pr+2.8/1000.0
Ray = (5.0 / 2.0)-ThicMaterial/2.0
EpaisCoque = 0.033
TassEp = 45
SphBaseCut = 0.5

#----------- config section
# TODO_LATER : to and from a file
Params={}
Params['PosVP'] = {} # (pointFocal) 4K UHD (youtube)
Params['PosVP']['Focal'] = ( 0.0, -(2.5+6.0), 3) # retrait de 6m a 3m de haut
#Params['PosVP']['Focal'] = ( 0.0, -(2.5+6.0), 4) # retrait de 6m a 4m de haut
#Params['PosVP']['Visee'] = ( 0.0, 0.0, (3.0-0.5)/2.0) # hauteur de visée à revoir suivant decoupe de sphere
Params['PosVP']['Visee'] = ( 0.0, 0.0, 0.5) # hauteur de visée à revoir suivant decoupe de sphere .. TODO_HERE something rotten, does not move as expected
Params['PosVP']['Up'] = ( 0.0, 0.0, 1.0) # Up vector
Params['PosVP']['d'] = 0.5 # ecran de coupe a 50 cm

Params['SourcePath']="../../../SourceMedia/"
Params['DestPath'] = "../../../ResultMedia/"

#Params['Source']="stsci-h-p1936b-f-3600x1800.jpg"
#Params['Source']="MOLA_mercat_redim.jpg"
#Params['Source']="MOLA_cylin.jpg"
#Params['Source']="2kjKZ.png"
Params['Source']="Vaches_de_race_montbéliarde.jpg"
#Params['Source']="Grid.png"

#Params['ResultSize'] = ( 3840, 2160) # (x,y) 4K UHD (youtube)
#Params['ResultSize'] = ( 1920, 1080) # (x,y) Full Hd
#Params['ResultSize'] = ( 1280,  720) # (x,y) HD Ready
Params['ResultSize'] = (  720,  480) # (x,y) DVD ... good enough for tests
#Params['ResultSize'] = (  360,  240) # SD
#Params['ResultSize'] = (  180,  120)
#Params['ResultSize'] = (  72,  48)

Params['AxisTilt'] = 25.19*2*Pi/360 # mars tilt axis

Params['ResultFps'] = 25
Params['ResultDuration'] = 4*60 # 4 min long video, accelerate on video edition if required
Params['dAlpha'] = -(2.0*Pi)/(Params['ResultFps']*Params['ResultDuration']*1.0) # counterclockwise rotation like earth  and most planets except venus, uranus, neptune

#----------- math section

def Dist(x, y, z):
    """
    distance entre un point et le centre
    """
    return(math.sqrt(x * x + y * y + z * z))

def VectSetNorm(V, N=1):
   """
   retaille le vecteur
   """
   (x, y, z) = V
   d = Dist(x, y, z)
   if d < Epsilon:
       d = Epsilon
       printf("Warn - too small vector to renorm")
   return(x * N / d, y * N / d, z * N / d)

def VectMultiply(V, N):
   """
   retaille le vecteur
   """
   ( x, y, z) = V
   return( x * N , y * N, z * N)

def AngulToScal(Alpha, Beta, R=1):
    """
    coordonnees angulaires (alpha, beta, r)
    Alpha - autour de l'axe z, direction x
    Beta - par rapport au plan x,y
    vers scalaires ( x, y, z)
    """
    return(R * math.cos(Beta) * math.cos(Alpha)
          , R * math.cos(Beta) * math.sin(Alpha)
          , R * math.sin(Beta)
          )

def ScalToAngul( Pt, Plane=None):
    """ polar coordinates from center (0,0,0) X direction (1,0,0) in 0,0 """
    (x,y,z) = Pt
    if Plane is None:
        T = ( x, y, 0)
        #T = PerpendiculairePointPlan( Pt, ( (0,0,0), (1,0,0), (0,1,0) ))
    else:
        T = PerpendiculairePointPlan( Pt, Plane)
        Ht = ProdVectoriel( Vector(Plane[0], Plane[1]), Vector(Plane[0], Plane[2]))
    if (Norm(T) <= Epsilon):
        Alpha = 0
        Beta = Pi/2.0
    else:
        if Plane is None:
            Alpha = VectorAngle( (1.0,0.0,0.0), T)
            Beta = VectorAngle(Pt, T)
            if (T[1] < 0):
                Alpha = 2.0 * Pi - Alpha
            if (Pt[2] < 0):
                Beta = -Beta
        else:
            Alpha = VectorAngle(Plane[1], T)
            Beta = VectorAngle( Pt, T)
            if (PerpendiculaireKPointDroite( Pt, (Plane[0], Plane[2])) < 0):
                Alpha = 2.0 * Pi - Alpha
            if (PerpendiculaireKPointDroite(Pt, (Plane[0], AddScal( Plane[0], Ht))) < 0): # TODO_HERE : as usual check direction
                Beta = -Beta
    R = Norm(Pt)
    return( Alpha, Beta, R)


def ProdScalaire(V1, V2):
   """
   Produit scalaire de deux vecteurs
   """
   (Vx1, Vy1, Vz1) = V1
   (Vx2, Vy2, Vz2) = V2
   return(Vx1 * Vx2 + Vy1 * Vy2 + Vz1 * Vz2)


def ProdVectoriel(V1, V2):
   """
   Produit vectoriel (de deux vecteurs)
   """
   (ax, ay, az) = V1
   (bx, by, bz) = V2
   Mx = (ay * bz) - (az * by)
   My = (az * bx) - (ax * bz)
   Mz = (ax * by) - (ay * bx)
   return(Mx, My, Mz)


def Norm(V1):
   """
   Norme du vecteur
   """
   (Vx1, Vy1, Vz1) = V1
   return(Dist(Vx1, Vy1, Vz1))


def Vector(Pt1, Pt2):
   """
   Construit un vecteur par deux points
   """
   # try:
   #   (Px1, Py1, Pz1) = Pt1
   # except :
   #   print "Debug"
   #   print "Pt1"
   #   print Pt1
   (Px1, Py1, Pz1) = Pt1
   (Px2, Py2, Pz2) = Pt2
   return(Px2 - Px1, Py2 - Py1, Pz2 - Pz1)

def AddScal(Pt1, Pt2):
   """
   Construit un point par un vecteur et un points
   """
   (Px1, Py1, Pz1) = Pt1
   (Px2, Py2, Pz2) = Pt2
   return(Px2 + Px1, Py2 + Py1, Pz2 + Pz1)


def VectorAngle(V1, V2):
   """
   Angle fait par deux vecteurs
   """
   N = Norm(V1) * Norm(V2)
   if (abs(N) <= Epsilon):
       #print("ERROR: VectorAngle - div par vecteur nul interdit!")
       #return("ERROR: VectorAngle - div par vecteur nul interdit!")
       return(0) # somehow aligned :) frequent in polar coordinate conversion
   PScal = ProdScalaire(V1, V2) / N
   if ((PScal > -1.00001) & (PScal < -1.)):
       PScal = -1.
   if ((PScal > 1.) & (PScal < 1.00001)):
       PScal = 1.
   if ((PScal < -1.) | (PScal > 1.)):
       print("ERROR: VectorAngle - math error")
       return("ERROR : VectorAngle - MathError")
   return(math.acos(PScal))


def IntersecPlanDroite(PA, DA):
   """
   Point d'intersection entre un plan et une droite
   Attention : cette version ne supporte pas tous les cas de vecteurs paralleles au repere
   """
   (Pt1, Pt2, Pt3) = PA
   (D1, D2) = DA
   
   def Cross4(KA, RA):
      (k1, k2, k3, k4) = KA
      (r1, r2, r3, r4) = RA
      return((k1 * r4) - (r1 * k4), (k2 * r4) - (r2 * k4), (k3 * r4) - (r3 * k4))
      
   def Cross3(KA, TA):
      (s1, s2, s3) = KA
      (t1, t2, t3) = TA
      return((s1 * t3) - (t1 * s3), (s2 * t3) - (t2 * s3))

   # (Pt1, Pt2, Pt3) = Plan
   (Px1, Py1, Pz1) = Pt1
   (Px2, Py2, Pz2) = Pt2   
   (Px3, Py3, Pz3) = Pt3
   # (D1, D2) = Droite
   (Dx1, Dy1, Dz1) = D1   
   (Dx2, Dy2, Dz2) = D2

   #                    D1P1     D2D1     P1P2     P1P3
   (X1, X2, X3, X4) = (Px1 - Dx1, Dx1 - Dx2, Px2 - Px1, Px3 - Px1)
   (Y1, Y2, Y3, Y4) = (Py1 - Dy1, Dy1 - Dy2, Py2 - Py1, Py3 - Py1)
   (Z1, Z2, Z3, Z4) = (Pz1 - Dz1, Dz1 - Dz2, Pz2 - Pz1, Pz3 - Pz1)

   (T1, T2, T3) = Cross4((X1, X2, X3, X4), (Z1, Z2, Z3, Z4))
   (U1, U2, U3) = Cross4((Y1, Y2, Y3, Y4), (Z1, Z2, Z3, Z4))
   (W1a, W2a) = Cross3((T1, T2, T3), (U1, U2, U3))

   (T1, T2, T3) = Cross4((X1, X2, X3, X4), (Y1, Y2, Y3, Y4))
   (U1, U2, U3) = Cross4((Z1, Z2, Z3, Z4), (Y1, Y2, Y3, Y4))
   (W1b, W2b) = Cross3((T1, T2, T3), (U1, U2, U3))

   (T1, T2, T3) = Cross4((Y1, Y2, Y3, Y4), (X1, X2, X3, X4))
   (U1, U2, U3) = Cross4((Z1, Z2, Z3, Z4), (X1, X2, X3, X4))
   (W1c, W2c) = Cross3((T1, T2, T3), (U1, U2, U3))

   if (abs(W2a) > abs(W2b)):
      if(abs(W2a) > abs(W2c)):
          (W1, W2) = (W1a, W2a)
      else:
          (W1, W2) = (W1c, W2c)
   else:       
      if(abs(W2b) > abs(W2c)):
          (W1, W2) = (W1b, W2b)
      else:
          (W1, W2) = (W1c, W2c)

   try:
      k = W1 / W2
      return(Dx1 + k * X2, Dy1 + k * Y2, Dz1 + k * Z2)
   except :
      # error tant que les utilisateurs seront pas blindes contre les reponses nulles
      # print "ERROR : IntersecPlanDroite : vecteurs colineaires"
      # print((Pt1, Pt2, Pt3), (D1, D2))
      return(Dx1, Dy1, Dz1)
      raise


def SolveAx2bxc(a, b, c):
   if (a == 0):
      if (b == 0):
         # c = 0... huuu c pas possib sauf si
         if (c == 0):
            # 0 = 0...
            return(0)
         else :
            # c = 0
            return()
      else :
         # bx+c = 0
         return(-c / b)
   else :
      D = ((b * b) - (4 * a * c))
      if(D == 0):
         return((-b / (2 * a)),)
      if(D >= 0):
         return((-b + math.sqrt(D)) / (2 * a), (-b - math.sqrt(D)) / (2 * a))
      else :
         return()


def PerpendiculaireKPointDroite(Pt1, Droite):
   """
   Resutltat k, D1+kD1D2 projection perpendiculaire de Pt1 sur la droite D1D2
   """
   (Px1, Py1, Pz1) = Pt1
   (D1, D2) = Droite
   (Dx1, Dy1, Dz1) = D1
   (Dx2, Dy2, Dz2) = D2
   X1 = Dx2 - Dx1
   Y1 = Dy2 - Dy1
   Z1 = Dz2 - Dz1
   XP1 = Px1 - Dx1
   YP1 = Py1 - Dy1
   ZP1 = Pz1 - Dz1
   k = (XP1 * X1 + YP1 * Y1 + ZP1 * Z1) / (X1 * X1 + Y1 * Y1 + Z1 * Z1)
   return(k)

def PerpendiculairePointDroite(Pt1, Droite):
   """
   Resutltat H, projection perpendiculaire de Pt1 sur la droite
   """
   (Px1, Py1, Pz1) = Pt1
   (D1, D2) = Droite
   (Dx1, Dy1, Dz1) = D1
   (Dx2, Dy2, Dz2) = D2
   X1 = Dx2 - Dx1
   Y1 = Dy2 - Dy1
   Z1 = Dz2 - Dz1
   XP1 = Px1 - Dx1
   YP1 = Py1 - Dy1
   ZP1 = Pz1 - Dz1
   k = (XP1 * X1 + YP1 * Y1 + ZP1 * Z1) / (X1 * X1 + Y1 * Y1 + Z1 * Z1)
   return(Dx1 + k * X1, Dy1 + k * Y1, Dz1 + k * Z1)


def PerpendiculairePointPlan(T1, Plan):
   """
   Resutltat H, projection perpendiculaire de Pt1 sur le plan
   """
   (P1, P2, P3) = Plan 
   Norm = ProdVectoriel(Vector(P1, P2), Vector(P1, P3))
   try :
      M = IntersecPlanDroite(Plan, (T1, AddScal(T1, Norm)))
      return(M)
   except :
      print("ERROR : intersec PointPlan")
      return()

def Map( V, S1, S2, D1, D2):
    """
    map V [S1, S2] to [D1, D2]
    """
    if (abs(S2 - S1) <= Epsilon):
       # print( (S1, S2, D1, D2))
       a = 1
       b = (D1 + D2) / 2.
    else :
       a = (D2 - D1) / (S2 - S1)
       b = D1 - a * S1
    return(a*V+b)


def StrDuration( Seconds):
    """ convert seconds in something a human can apprehend"""
    OrgSec = Seconds
    Str = ""
    St = False
    Seg = 3600*24 # j
    if (Seconds > Seg or St):
        Tmp = math.floor( Seconds / Seg)
        Str += "%ij" % Tmp
        Seconds -= Tmp* Seg
        St = True
    Seg = 3600 # H
    if (Seconds > Seg or St):
        Tmp = math.floor( Seconds / Seg)
        Str += "%2.2iH" % Tmp
        Seconds -= Tmp* Seg
        St = True
    Seg = 60 # min
    if (Seconds > Seg or St):
        Tmp = math.floor( Seconds / Seg)
        Str += "%2.2i:" % Tmp
        Seconds -= Tmp* Seg
        St = True
    Seg = 1
    if(1):
        Tmp = math.floor( Seconds / Seg)
        if (St):
            Str += "%2.2is" % Tmp
        else:
            Str += "%is" % Tmp
        Seconds -= Tmp* Seg
    if (St):
        Str += "(%is)" % OrgSec

    return( Str)
#--------------- projecteur
# le plan de projection c'est un point central un vecteurs droite et un vecteur haut

VPFocal = Params['PosVP']['Focal']
VTmp = Vector( VPFocal, Params['PosVP']['Visee'])
VPPtCenter = AddScal( VPFocal, VectSetNorm( VTmp, Params['PosVP']['d']))
VPUp = Params['PosVP']['Up']
VPVectDroite = ProdVectoriel( VPUp, VPPtCenter)
VPVectDroite = VectSetNorm( VPVectDroite) # vect d'1 metre
VpPtDroite = AddScal( VPPtCenter, VPVectDroite)
VPVectUp = ProdVectoriel( VPPtCenter, VPVectDroite)
VPVectUp = VectSetNorm( VPVectUp)
VpPtUp = AddScal( VPPtCenter, VPVectUp)

#----------- image manip
#image = ImageSource.open("C:/Users/emman/Google Drive/Op/SphereByTriangle/Projection/SourceMedia/stsci-h-p1936b-f-3600x1800.jpg")
ImageSource = Image.open(Params['SourcePath']+Params['Source'])
ImageSourcePx = ImageSource.load()
ImageSourceSize = ImageSource.size

ImageResult=Image.new('RGB', (Params['ResultSize'][0], Params['ResultSize'][1]), (0, 0, 0)) # prepare an all black image
ImageResultDraw = ImageDraw.Draw(ImageResult)

# before auto adjust
ResultF = 1
ResultDx = 0
ResultDy = 0

def Project2D( Pt3D):
    """ 2D screen coordinate projection in pixels of a point of the 3D scene """
    PtScr = IntersecPlanDroite((VPPtCenter, VpPtDroite, VpPtUp), (VPFocal, Pt3D))
    PtX = PerpendiculaireKPointDroite(PtScr, (VPPtCenter, VpPtDroite))
    PtY = PerpendiculaireKPointDroite(PtScr, (VPPtCenter, VpPtUp))
    PtX = ResultF*PtX + ResultDx
    PtY = -ResultF*PtY + ResultDy

    return( PtX, PtY)

def SourceGetColor( Alpha, Gamma, Beta, Delta, Tilt = 0):
    """ Get color under a point given it's polar coordinates on the sphere
        (Alpha+Gamma, Beta) - coordinates to map on the ... map
        Gamma is rotation around the axis
        Tilt inclination of the axis around Y
        Delta - the size of the dot (or an indication of)  """
    if (Delta < 0):
        Delta = 0
    if (0 != Tilt):
        Pt = AngulToScal( Alpha, Beta, )
        (Alpha, Beta, Rtmp) = ScalToAngul( Pt, ( (0,0,0), (math.cos(Tilt),0,-math.sin(Tilt)), (0,1,0) ))
    (Alpha, Beta) = AngulModulus(Alpha + Gamma, Beta)

    X = Map( Alpha, 0, 2*Pi, 0, ImageSourceSize[0])
    Y = Map( Beta , Pi/2.0, -Pi/2.0, 0, ImageSourceSize[1])
    X2 = Map( Alpha+Delta, 0, 2*Pi, 0, ImageSourceSize[0])
    Y2 = Map( Beta+Delta , Pi/2.0, -Pi/2.0, 0, ImageSourceSize[1])
    dX = abs( X-X2)
    dY = abs( Y-Y2)
    QColor = 0.0
    Cr = 0
    Cg = 0
    Cb = 0
    Lim = 5
    if (dX > 2 and dY > 2):
        if(dX > Lim):
            Px = dX/Lim
        else:
            Px = 1
        if(dY > Lim):
            Py = dY/Lim
        else:
            Py = 1
        DQ = 0
        Mx = X
        while(Mx<X+dX and Mx<ImageSourceSize[0]):
            My = Y
            while(My < Y+dY and My<ImageSourceSize[1]):
                Color2 = ImageSource.getpixel((Mx, My))
                Cr += Color2[0]
                Cg += Color2[1]
                Cb += Color2[2]
                QColor +=1
                My = My+Py
            Mx = Mx + Px
        if (QColor>= 1):
            Color = (Cr/QColor,Cg/QColor,Cb/QColor)
        else:
            Color = ImageSource.getpixel((X, Y))
    elif ((dX+dY) < 0.5):
        Color = ImageSource.getpixel((X, Y))
    else:
        Color = ImageSource.getpixel( (X,Y))
    return(  Color)

def IntersecDroiteSphereNear( VPFocal, PtScr):
    P1P2 = Vector( VPFocal, PtScr)
    P1O = Vector(VPFocal, ( 0.0, 0.0, 0.0))
    a = ProdScalaire( P1P2, P1P2)
    b = 0.0-2.0*ProdScalaire( P1P2, P1O)
    c = ProdScalaire( P1O, P1O)-Ray*Ray
    Sol = SolveAx2bxc( a, b, c)
    if (0 == len(Sol)):
        return()
    if (1 == len(Sol)):
        k = Sol[0]
    if (2 == len(Sol)):
        if (Sol[1] < Sol[0]):
            k = Sol[1]
        else:
            k = Sol[0]
    PtSph = AddScal( VPFocal, VectMultiply( Vector(VPFocal, PtScr), k))
    (Alpha, Beta, Rtmp) = ScalToAngul(PtSph)
    if (Beta < BetaBase):
        return()

    return( (Alpha,Beta))

def AngulModulus( Alpha, Beta):
    """ rebound angles"""
    Alpha = Alpha % (2.0*Pi)
    if(Beta < -Pi/2.0):
        Beta  = -Pi/2.0
    if(Beta > Pi/2.0):
        Beta  = Pi/2.0
    return( (Alpha,Beta))

def ColorMix( C1, v1, C2, v2):
    """ Mix 2 colors  with some kind of barycenter """
    V12=v1+v2
    return( ( (C1[0]*v1+C2[0]*v2)/V12
            , (C1[1]*v1+C2[1]*v2)/V12
            , (C1[2]*v1+C2[2]*v2)/V12
          ) )

def ColorFix(C1):
    """ because real color does not like floats and must be round """
    return( ( int(C1[0]), int(C1[1]), int(C1[2]) ) )

# automatic scale for picture to fit the destination image
PtSph = AngulToScal(0, 0, Ray)
(PtX, PtY) = Project2D( PtSph)

F = abs(PtX)
F1= abs(PtY)
if (F1 > F):
    F = F1

Mx = F*0.90 # length of VPVectDroite mapped to the screen
ResultF = (Params['ResultSize'][1]/2.0) / Mx # en pix/m
ResultDx = Params['ResultSize'][0]/2.0
ResultDy = Params['ResultSize'][1]/2.0

BetaBase = -math.asin(SphBaseCut/Ray)

AddAlpha = 0
ImgR0 = None
BackgroundColor = ( 0, 0, 0)

def DirectTrace( GammaRotation):
    global pict_time
    Tilt = Params['AxisTilt']
    dt_start_time = time.time()
    Segs = 50.0
    Beta=BetaBase
    while ( Beta < Pi/2.0):
        DBet = Pi/Segs
        Beta2 = Beta + DBet
        Alp = Pi
        while( Alp < 2*Pi):
            DAlph = Pi/(2*Segs)
            if ( Alp < Pi): # draw left to middle then right to middle
                Alpha = Pi
            else:
                Alpha = -Alp+3.0*Pi

            Color = SourceGetColor( Alpha, GammaRotation, Beta, (DBet+DAlph)/2.0, Tilt)

            PtSph00 = AngulToScal( Alpha, Beta, Ray)
            PtSph01 = AngulToScal( Alpha, Beta+DBet, Ray)
            PtSph11 = AngulToScal( Alpha+DAlph, Beta+DBet, Ray)
            PtSph10 = AngulToScal( Alpha+DAlph, Beta, Ray)
            #Chk = ScalToAngul(PtSph)
            #if ( abs(Chk[0]-Alpha) > Epsilon):
            #    print("Bad Alpha");
            #if ( abs(Chk[1]-Beta) > Epsilon):
            #    print("Bad Beta");

            (PtX00, PtY00) = Project2D( PtSph00)
            (PtX01, PtY01) = Project2D( PtSph01)
            (PtX11, PtY11) = Project2D( PtSph11)
            (PtX10, PtY10) = Project2D( PtSph10)
            #ImageResultDraw.line( (PtX00, PtY00, PtX11, PtY11), fill=Color, width=5)
            ImageResultDraw.polygon(  (PtX00, PtY00, PtX01, PtY01, PtX11, PtY11, PtX10, PtY10) , fill=ColorFix(Color)) # TODO_HERE : test

            Alp = Alp + DAlph
        Beta = Beta2
    pict_time = time.time() - dt_start_time
    #ImageResult.show()

def OneRay( Vx, Vy):
    """ Get (Alpha, Beta) on the sphere for a given fraction of right and up Vectoss"""
    VScrX = VectSetNorm(VPVectDroite, Vx)  # Y is the ref to keep square pix
    VScrY = VectSetNorm(VPVectUp, Vy)
    PtScr = AddScal(AddScal(VPPtCenter, VScrX), VScrY)
    Res = IntersecDroiteSphereNear(VPFocal, PtScr)
    return( Res)

def RayTrace( GammaRotation):
    global pict_time
    global ImgR0
    Tilt = Params['AxisTilt']
    rt_start_time = time.time()
    Sth = 0
    X = 0
    while(X < 2*ResultDx):
        Vx = Map(X, 0, 2*ResultDx, -Mx*ResultDx/ResultDy, Mx*ResultDx/ResultDy)
        Vx2 = Map(X+1, 0, 2*ResultDx, -Mx*ResultDx/ResultDy, Mx*ResultDx/ResultDy)
        Y = 0
        while(Y < 2*ResultDy):
            if (X==ResultDx and Y==ResultDy):
                #ImageResult.show()
                print("Mid", end='', flush=True)
            Vy = Map(Y, 0, 2*ResultDy, Mx, -Mx)
            Vy2 = Map(Y+1, 0, 2*ResultDy, Mx, -Mx)
            if (1): # TODO_LATER : first test it acts the same and not too slow
                Res = OneRay( Vx, Vy)
            else :
                VScrX = VectSetNorm( VPVectDroite, Vx) # Y is the ref to keep square pix
                VScrY = VectSetNorm(VPVectUp, Vy)
                PtScr = AddScal( AddScal( VPPtCenter, VScrX), VScrY)
                Res = IntersecDroiteSphereNear( VPFocal, PtScr)
            D = -1
            if (len(Res) >= 2):
                if (1):
                    Res2 = OneRay(Vx2, Vy2)
                    if (len(Res2) >= 2):
                        D = (abs(Res[0] - Res2[0]) + abs(Res[1] - Res2[1])) / 2.0
                Color = SourceGetColor( Res[0], GammaRotation, Res[1], D, Tilt)
                #ImageResult.putpixel( X, Y, Color)
                if( None != ImgR0):
                    Color2 = ImgR0.getpixel( (X,Y))
                    Color = ColorFix( ColorMix ( Color, 1, Color2, -1))
                ImageResultDraw.point( (X, Y), ColorFix(Color))
                #ImageResultDraw.line((X, Y, X + 1, Y + 1), fill=Color, width=2)
                Sth = Sth + 1
            else:
                ImageResultDraw.point( (X, Y), BackgroundColor)
                Sth = Sth + 0
            Y = Y+1
        if (Sth > 100000):
            #ImageResult.show()
            print(".", end='', flush=True)
            Sth = 0
        X = X+1
    # ImgR0 = ImageResult.copy() # moyenner evite 50% le grain mais pas suffisant pour faire le flou de mouvement
    #print("-")
    pict_time = time.time() - rt_start_time


#RayTrace( AddAlpha)

def PictAndVid( ImgIdxStart, ImgModul):
    ImgsMax = Params['ResultFps'] * Params['ResultDuration']
    dAlpha = Params['dAlpha']
    outvid = Params['DestPath']+"result.avi"
    fps = Params['ResultFps']
    pict_time_R0 = None
    ImgVidIdx = 0

    vid = None
    if (0 == ImgIdxStart):
        size = None
        is_color = True
        format = "XVID"
        fourcc = cv2.VideoWriter_fourcc(*format)

    ImgIdx = ImgIdxStart
    while(ImgIdx < ImgsMax):
        AddAlpha = ImgIdx * dAlpha
        #DirectTrace( AddAlpha)
        RayTrace( AddAlpha)
        ImageResult.save(Params['DestPath']+'Result_%i.jpeg' % ImgIdx, 'jpeg')
        ImgIdx += 1

        if (None == pict_time_R0):
            pict_time_R0 = pict_time
        pict_time_R0 = (pict_time_R0*3+pict_time)/4
        print("pr %i Image %i/%i %2.2f%% r%f takes %s possib remains %s" % ( ImgIdxStart, ImgIdx, ImgsMax, 100*ImgIdx/(ImgsMax*1.0), AddAlpha*360/(2.0*Pi)
                                                               , StrDuration (pict_time), StrDuration ((ImgsMax-ImgIdx)*pict_time_R0/ImgModul)))

        if (ImgModul > 1):
            ImgIdx += (ImgModul - 1)
        # http://www.xavierdupre.fr/blog/2016-03-30_nojs.html
        # http://www.fourcc.org/codecs.php
        # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
        if (0 == ImgIdxStart): # only for process 1
            ImageResult.save(Params['DestPath']+'Result.jpeg', 'jpeg') # must see intermediate result
            while (ImgVidIdx < ImgIdx and ImgVidIdx < ImgsMax):
                Found = False
                img = None
                while(not Found):
                    try:
                        img = cv2.imread(Params['DestPath']+'Result_%i.jpeg' % ImgVidIdx)
                        if (not( img is None)):
                            Found = True
                        else:
                            print( "Sleep a while to wait for pic %i" % ImgVidIdx);
                            time.sleep(4)
                    except:
                        print( "Sleep a while to wait for pic %i" % ImgVidIdx);
                        time.sleep(4)
                if vid is None:
                    if size is None:
                        size = (img.shape[1], img.shape[0])
                    vid = cv2.VideoWriter( outvid, fourcc, float(fps), size, is_color)
                if size[0] != img.shape[1] and size[1] != img.shape[0]:
                    img = cv2.resize(img, size)
                vid.write(img)
                os.remove("%sResult_%i.jpeg" % (Params['DestPath'], ImgVidIdx))
                print("Integrated pic %i" % ImgVidIdx);
                ImgVidIdx += 1
    if not(vid is None):
        vid.release()
    #----------- Produce result
    #ImageResult.show()


def ProcessMe( PrNum):
    if(0 == PrNum):
        print( "I'll Own Video")
        PictAndVid( PrNum, Processes)
    else:
        print( "I'm process %i" % PrNum)
        PictAndVid( PrNum, Processes)


if __name__ == '__main__':
    FileList = glob.glob("%sResult_*.jpeg" % (Params['DestPath']))
    for filePath in FileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    if (Processes>1):
        ListArgs=[0,]
        Idx = 1
        while( Idx < Processes):
            ListArgs = ListArgs + [Idx,]
            Idx += 1
        with Pool(Processes) as p:
            p.map(ProcessMe, ListArgs)
    else:
        ProcessMe(0)
    print( "Done in %s" % StrDuration(time.time() - start_time))
