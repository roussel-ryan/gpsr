import numpy as np

def drift(L):
    D = np.identity(6)
    D[0,1] = L
    D[2,3] = L
    return D

def bend_face(h, beta):
    face = np.identity(6)
    face[1,0] = h*np.tan(beta)
    face[3,2] = -h*np.tan(beta)
    return face

def bend_body(L, alpha):
    h = alpha/L
    body = np.identity(6)
    C = np.cos(alpha)
    S = np.sin(alpha)
    body[0,0] = C
    body[0,1] = S/h
    body[0,5] = (1-C)/h
    body[1,0] = -h*S
    body[1,1] = C
    body[1,5] = S
    body[2,3] = L
    body[4,0] = S
    body[4,1] = (1-C)/h
    body[4,5] = (alpha-S/h)
    return body

def bend(L, alpha, beta1, beta2):
    alpha_new = -alpha
    beta1_new = -beta1
    beta2_new = -beta2
    h = alpha_new/L
    entrance = bend_face(h, beta1_new)
    body = bend_body(L, alpha_new)
    exit = bend_face(h, beta2_new)
    return exit @ body @ entrance
