import jittor as jt
import numpy as np
import math
import pdb

EYE_H = 40
EYE_W = 56
NOSE_H = 48
NOSE_W = 48
MOUTH_H = 56
MOUTH_W = 64

def masked(A, mask):
    return (A/2+0.5)*mask*2-1

def inverse_mask(mask):
    return jt.ones(mask.shape)-mask

def addone_with_mask(A, mask):
    return ((A/2+0.5)*mask + (jt.ones(mask.shape)-mask))*2-1

def add_with_mask(A, B, mask):
    return ((A/2+0.5)*mask + (B/2+0.5)*(jt.ones(mask.shape)-mask))*2-1

def partCombiner2_bg(center, eyel, eyer, nose, mouth, hair, bg, maskh, maskb, comb_op = 1, load_h = 512, load_w = 512):
    if comb_op == 0:
        # use max pooling, pad black for eyes etc
        padvalue = -1
        hair = masked(hair, maskh)
        bg = masked(bg, maskb)
    else:
        # use min pooling, pad white for eyes etc
        padvalue = 1
        hair = addone_with_mask(hair, maskh)
        bg = addone_with_mask(bg, maskb)
    ratio = load_h // 256
    rhs = np.array([EYE_H,EYE_H,NOSE_H,MOUTH_H]) * ratio
    rws = np.array([EYE_W,EYE_W,NOSE_W,MOUTH_W]) * ratio
    bs,nc,_,_ = eyel.shape
    eyel_p = jt.ones((bs,nc,load_h,load_w))
    eyer_p = jt.ones((bs,nc,load_h,load_w))
    nose_p = jt.ones((bs,nc,load_h,load_w))
    mouth_p = jt.ones((bs,nc,load_h,load_w))
    locals = [eyel, eyer, nose, mouth]
    locals_p = [eyel_p, eyer_p, nose_p, mouth_p]
    for i in range(bs):
        c = center[i].data#x,y
        for j in range(4):
            locals_p[j][i] = jt.nn.ConstantPad2d((int(c[j,0]-rws[j]/2), int(load_w-(c[j,0]+rws[j]/2)), int(c[j,1]-rhs[j]/2), int(load_h-(c[j,1]+rhs[j]/2))),padvalue)(locals[j][i])
    if comb_op == 0:
        eyes = jt.maximum(locals_p[0], locals_p[1])
        eye_nose = jt.maximum(eyes, locals_p[2])
        eye_nose_mouth = jt.maximum(eye_nose, locals_p[3])
        eye_nose_mouth_hair = jt.maximum(hair, eye_nose_mouth)
        result = jt.maximum(bg, eye_nose_mouth_hair)
    else:
        eyes = jt.minimum(locals_p[0], locals_p[1])
        eye_nose = jt.minimum(eyes, locals_p[2])
        eye_nose_mouth = jt.minimum(eye_nose, locals_p[3])
        eye_nose_mouth_hair = jt.minimum(hair, eye_nose_mouth)
        result = jt.minimum(bg, eye_nose_mouth_hair)
    return result

def getLocalParts(fakeAB, center, maskh, maskb, load_h = 512, load_w = 512):
    bs,nc,_,_ = fakeAB.shape
    ratio = load_h // 256
    rhs = np.array([EYE_H,EYE_H,NOSE_H,MOUTH_H]) * ratio
    rws = np.array([EYE_W,EYE_W,NOSE_W,MOUTH_W]) * ratio
    eyel = jt.ones((bs,nc,int(rhs[0]),int(rws[0])))
    eyer = jt.ones((bs,nc,int(rhs[1]),int(rws[1])))
    nose = jt.ones((bs,nc,int(rhs[2]),int(rws[2])))
    mouth = jt.ones((bs,nc,int(rhs[3]),int(rws[3])))
    locals = [eyel, eyer, nose, mouth]
    for i in range(bs):
        c = center[i].data
        for j in range(4):
            locals[j][i] = fakeAB[i, :, int(c[j,1]-rhs[j]//2):int(c[j,1]+rhs[j]//2), int(c[j,0]-rws[j]//2):int(c[j,0]+rws[j]//2)]
    hair = masked(fakeAB, maskh)
    bg = masked(fakeAB, maskb)
    locals += [hair, bg]
    return locals

def getonehot(outputs, classes, batch_size):
    index = jt.argmax(outputs,1)
    y = jt.unsqueeze(index[0],1)
    onehot = jt.zeros([batch_size, classes])
    onehot.scatter_(1, y, jt.array(1.))
    return onehot 

def nonlinearDt(dt, c=3.3):
    nldt = (jt.exp(dt*c)-1) / (jt.exp(c)-1)
    return nldt

W = 11
aa = int(math.floor(512./W))
res = 512 - W*aa
def get_patches(fake_B, face_mask):
    # [1,1,512,512]->[bs,1,11,11]
    patches = []
    weights = []
    W2 = int(W/2)
    t = np.random.randint(res,size=2)
    for i in range(aa):
        for j in range(aa):
            p = fake_B[:,:,t[0]+i*W:t[0]+(i+1)*W,t[1]+j*W:t[1]+(j+1)*W]
            whitenum = jt.sum(p>=0.0)
            if whitenum < 1 or whitenum > W*W-1:
                continue
            patches.append(p)
            weights.append(face_mask[:,:,t[0]+i*W+W2,t[1]+j*W+W2])
    fake_B_patches = jt.contrib.concat(patches, dim=0)
    conti_weights = jt.contrib.concat(weights, dim=0)+1 #0->1,1->2

    return fake_B_patches, conti_weights