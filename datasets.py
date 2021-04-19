import glob
import random
import os
import numpy as np

import jittor as jt
from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from PIL import Image
import csv
import random
import cv2

EYE_H = 40
EYE_W = 56
NOSE_H = 48
NOSE_W = 48
MOUTH_H = 56
MOUTH_W = 64

def getfeats(featpath):
	trans_points = np.empty([5,2],dtype=np.int64) 
	with open(featpath, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		for ind,row in enumerate(reader):
			trans_points[ind,:] = row
	return trans_points

def tocv2(ts):
    img = (ts.numpy()/2+0.5)*255
    img = img.astype('uint8')
    img = np.transpose(img,(1,2,0))
    img = img[:,:,::-1]#rgb->bgr
    return img

def dt(img):
    if(img.shape[2]==3):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #convert to BW
    ret1,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret2,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    dt1 = cv2.distanceTransform(thresh1,cv2.DIST_L2,5)
    dt2 = cv2.distanceTransform(thresh2,cv2.DIST_L2,5)
    dt1 = dt1/dt1.max()#->[0,1]
    dt2 = dt2/dt2.max()
    return dt1, dt2

def getSoft(size,xb,yb,boundwidth=5.0):
    xarray = np.tile(np.arange(0,size[1]),(size[0],1))
    yarray = np.tile(np.arange(0,size[0]),(size[1],1)).transpose()
    cxdists = []
    cydists = []
    for i in range(len(xb)):
        xba = np.tile(xb[i],(size[1],1)).transpose()
        yba = np.tile(yb[i],(size[0],1))
        cxdists.append(np.abs(xarray-xba))
        cydists.append(np.abs(yarray-yba))
    xdist = np.minimum.reduce(cxdists)
    ydist = np.minimum.reduce(cydists)
    manhdist = np.minimum.reduce([xdist,ydist])
    im = (manhdist+1) / (boundwidth+1) * 1.0
    im[im>=1.0] = 1.0
    return im


def get_transform(params, gray = False, mask = False):
    transform_ = []
    # resize
    transform_.append(transform.Resize((params['load_h'], params['load_w']), Image.BICUBIC))
    # flip
    if params['flip']:
        transform_.append(transform.Lambda(lambda img: transform.hflip(img)))
    if gray:
        transform_.append(transform.Gray())
    if mask:
        transform_.append(transform.ImageNormalize([0.,], [1.,]))
    else:
        if not gray:
            transform_.append(transform.ImageNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        else:
            transform_.append(transform.ImageNormalize([0.5,], [0.5,]))
    return transform.Compose(transform_)

class ImageDataset(Dataset):
    def __init__(self, root, mode="train", load_h=512, load_w=512):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(root, mode, "img") + "/*.*"))
        self.lmdir = os.path.join(root, mode, "landmark")
        self.maskbgdir = os.path.join(root, mode, "mask_bg")
        self.maskfacedir = os.path.join(root, mode, "mask_face")
        self.cmaskdir = os.path.join(root, mode, "cmask_{}")
        self.set_attrs(total_len=len(self.files))
        self.load_h = load_h
        self.load_w = load_w

    def __getitem__(self, index):

        AB_path = self.files[index % len(self.files)]
        img = Image.open(AB_path)
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        flip = random.random() > 0.5

        params = {'load_h': self.load_h, 'load_w': self.load_w, 'flip': flip}
        transform_A = get_transform(params)
        transform_B = get_transform(params, gray=True)
        transform_mask = get_transform(params, gray=True, mask=True)

        item_A = transform_A(img_A)
        item_A = jt.array(item_A)
        item_B = transform_B(img_B)
        item_B = jt.array(item_B)

        item_A_l = {}
        item_B_l = {}
        regions = ['eyel','eyer','nose','mouth']
        basen = os.path.basename(AB_path)[:-4]
        lm_path = os.path.join(self.lmdir, basen+'.txt')
        feats = getfeats(lm_path)
        if flip:
            for i in range(5):
                feats[i,0] = self.load_w - feats[i,0] - 1
            tmp = [feats[0,0],feats[0,1]]
            feats[0,:] = [feats[1,0],feats[1,1]]
            feats[1,:] = tmp
        mouth_x = int((feats[3,0]+feats[4,0])/2.0)
        mouth_y = int((feats[3,1]+feats[4,1])/2.0)
        ratio = self.load_h // 256
        rhs = np.array([EYE_H,EYE_H,NOSE_H,MOUTH_H]) * ratio
        rws = np.array([EYE_W,EYE_W,NOSE_W,MOUTH_W]) * ratio
        center = np.array([[feats[0,0],feats[0,1]-4*ratio],[feats[1,0],feats[1,1]-4*ratio],[feats[2,0],feats[2,1]-rhs[2]//2+16*ratio],[mouth_x,mouth_y]])
        soft_border_mask4 = []
        for i in range(4):
            xb = [np.zeros(rhs[i]),np.ones(rhs[i])*(rws[i]-1)]
            yb = [np.zeros(rws[i]),np.ones(rws[i])*(rhs[i]-1)]
            soft_border_mask = getSoft([rhs[i],rws[i]],xb,yb)
            soft_border_mask = jt.array(soft_border_mask).unsqueeze(0).float()
            soft_border_mask4.append(soft_border_mask)
        for i in range(4):
            item_A_l[regions[i]+'_A'] = item_A[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)] * soft_border_mask4[i].repeat(3,1,1)
            item_B_l[regions[i]+'_B'] = item_B[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)] * soft_border_mask4[i]
        
        cmasks = []
        for i in range(4):
            if not flip or i not in [2,3]:
                cmaskpath = os.path.join(self.cmaskdir.format(regions[i]),basen+'.png')
            else:
                cmaskpath = os.path.join(self.cmaskdir.format(regions[1-i]),basen+'.png')
            im_cmask = Image.open(cmaskpath)
            cmask0 = transform_mask(im_cmask)
            cmask0 = jt.array(cmask0)
            cmask0 = (cmask0 >= 0.5).float()
            cmask = cmask0.clone()
            cmask = cmask[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)]
            cmasks.append(cmask)
        
        mask = jt.ones([1,item_A.shape[1],item_A.shape[2]]) # mask out eyes, nose, mouth
        for i in range(4):
            mask[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)] = 0
        imgsize = self.load_h
        maskn = mask[0].numpy()
        masks = [np.ones([imgsize,imgsize]),np.ones([imgsize,imgsize]),np.ones([imgsize,imgsize]),np.ones([imgsize,imgsize])]
        masks[0][1:] = maskn[:-1]
        masks[1][:-1] = maskn[1:]
        masks[2][:,1:] = maskn[:,:-1]
        masks[3][:,:-1] = maskn[:,1:]
        masks2 = [maskn-e for e in masks]
        bound = np.minimum.reduce(masks2)
        bound = -bound
        xb = []
        yb = []
        for i in range(4):
            xbi = [center[i,0]-rws[i]/2, center[i,0]+rws[i]/2-1]
            ybi = [center[i,1]-rhs[i]/2, center[i,1]+rhs[i]/2-1]
            for j in range(2):
                maskx = bound[:,int(xbi[j])]
                masky = bound[int(ybi[j]),:]
                tmp_a = maskx * xbi[j]
                tmp_b = 1-maskx
                xb += [tmp_b*10000 + tmp_a]

                tmp_a = masky * ybi[j]
                tmp_b = 1-masky
                yb += [tmp_b*10000 + tmp_a]
        soft = 1-getSoft([imgsize,imgsize],xb,yb)
        soft = jt.array(soft).unsqueeze(0).float()
        mask = (jt.ones(mask.shape)-mask)*soft + mask

        bgpath = os.path.join(self.maskbgdir, basen+'.png')
        im_bg = Image.open(bgpath)
        mask2 = transform_mask(im_bg) # mask out background
        mask2 = jt.array(mask2)
        mask2 = (mask2 >= 0.5).float() # foreground: 1, background: 0
        item_A_l['hair_A'] = (item_A/2+0.5) * mask.repeat(3,1,1) * mask2.repeat(3,1,1) * 2 - 1
        item_A_l['bg_A'] = (item_A/2+0.5) * (jt.ones(mask2.shape)-mask2).repeat(3,1,1) * 2 - 1
        item_B_l['hair_B'] = (item_B/2+0.5) * mask * mask2 * 2 - 1
        item_B_l['bg_B'] = (item_B/2+0.5) * (jt.ones(mask2.shape)-mask2) * 2 - 1

        facepath = os.path.join(self.maskfacedir, basen+'.png')
        im_face = Image.open(facepath)
        maskface = transform_mask(im_face)
        maskface = jt.array(maskface)

        img = tocv2(item_B)
        dt1, dt2 = dt(img)
        dt1 = jt.array(dt1)
        dt2 = jt.array(dt2)
        dt1 = dt1.unsqueeze(0)
        dt2 = dt2.unsqueeze(0)

        return item_A, item_A_l['eyel_A'], item_A_l['eyer_A'], item_A_l['nose_A'], item_A_l['mouth_A'], item_A_l['hair_A'], item_A_l['bg_A'], item_B, item_B_l['eyel_B'], item_B_l['eyer_B'], item_B_l['nose_B'], item_B_l['mouth_B'], item_B_l['hair_B'], item_B_l['bg_B'], mask, mask2, center, dt1, dt2, cmasks[0], cmasks[1], cmasks[2], cmasks[3], maskface

class TestDataset(Dataset):
    def __init__(self, root, lmdir, maskdir, cmaskdir, mode="test", load_h=512, load_w=512):
        super().__init__()
        transform_ = [
            transform.Resize((load_h, load_w), Image.BICUBIC),
            transform.ImageNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        self.transform = transform.Compose(transform_)
        transform_mask_ = [
            transform.Resize((load_h, load_w), Image.BICUBIC),
            transform.Gray(),
        ]
        self.transform_mask = transform.Compose(transform_mask_)

        self.files_A = sorted(glob.glob(root + "/*.*"))

        self.total_len = len(self.files_A)
        self.batch_size = None
        self.shuffle = False
        self.drop_last = False
        self.num_workers = None
        self.buffer_size = 512*1024*1024

        self.lmdir = lmdir
        self.maskdir = maskdir
        self.cmaskdir = cmaskdir
        self.load_h = load_h

    def __getitem__(self, index):
        A_path = self.files_A[index % len(self.files_A)]
        image_A = Image.open(A_path)

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)

        item_A = self.transform(image_A)
        item_A = jt.array(item_A)

        item_A_l = {}
        regions = ['eyel','eyer','nose','mouth']
        basen = os.path.basename(A_path)[:-4]
        lm_path = os.path.join(self.lmdir, basen+'.txt')
        feats = getfeats(lm_path)
        mouth_x = int((feats[3,0]+feats[4,0])/2.0)
        mouth_y = int((feats[3,1]+feats[4,1])/2.0)
        ratio = self.load_h // 256
        rhs = np.array([EYE_H,EYE_H,NOSE_H,MOUTH_H]) * ratio
        rws = np.array([EYE_W,EYE_W,NOSE_W,MOUTH_W]) * ratio
        center = np.array([[feats[0,0],feats[0,1]-4*ratio],[feats[1,0],feats[1,1]-4*ratio],[feats[2,0],feats[2,1]-rhs[2]//2+16*ratio],[mouth_x,mouth_y]])
        soft_border_mask4 = []
        for i in range(4):
            xb = [np.zeros(rhs[i]),np.ones(rhs[i])*(rws[i]-1)]
            yb = [np.zeros(rws[i]),np.ones(rws[i])*(rhs[i]-1)]
            soft_border_mask = getSoft([rhs[i],rws[i]],xb,yb)
            soft_border_mask = jt.array(soft_border_mask).unsqueeze(0).float()
            soft_border_mask4.append(soft_border_mask)
        for i in range(4):
            item_A_l[regions[i]+'_A'] = item_A[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)] * soft_border_mask4[i].repeat(3,1,1)
        
        cmasks = []
        for i in range(4):
            cmaskpath = os.path.join(self.cmaskdir.format(regions[i]),basen+'.png')
            im_cmask = Image.open(cmaskpath)
            cmask0 = self.transform_mask(im_cmask)
            cmask0 = jt.array(cmask0)
            cmask0 = (cmask0 >= 0.5).float()
            cmask = cmask0.clone()
            cmask = cmask[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)]
            cmasks.append(cmask)

        mask = jt.ones([1,item_A.shape[1],item_A.shape[2]]) # mask out eyes, nose, mouth
        for i in range(4):
            mask[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)] = 0
        imgsize = self.load_h
        maskn = mask[0].numpy()
        masks = [np.ones([imgsize,imgsize]),np.ones([imgsize,imgsize]),np.ones([imgsize,imgsize]),np.ones([imgsize,imgsize])]
        masks[0][1:] = maskn[:-1]
        masks[1][:-1] = maskn[1:]
        masks[2][:,1:] = maskn[:,:-1]
        masks[3][:,:-1] = maskn[:,1:]
        masks2 = [maskn-e for e in masks]
        bound = np.minimum.reduce(masks2)
        bound = -bound
        xb = []
        yb = []
        for i in range(4):
            xbi = [center[i,0]-rws[i]/2, center[i,0]+rws[i]/2-1]
            ybi = [center[i,1]-rhs[i]/2, center[i,1]+rhs[i]/2-1]
            for j in range(2):
                maskx = bound[:,int(xbi[j])]
                masky = bound[int(ybi[j]),:]
                tmp_a = maskx * xbi[j]
                tmp_b = 1-maskx
                xb += [tmp_b*10000 + tmp_a]

                tmp_a = masky * ybi[j]
                tmp_b = 1-masky
                yb += [tmp_b*10000 + tmp_a]
        soft = 1-getSoft([imgsize,imgsize],xb,yb)
        soft = jt.array(soft).unsqueeze(0).float()
        mask = (jt.ones(mask.shape)-mask)*soft + mask

        bgpath = os.path.join(self.maskdir, basen+'.png')
        im_bg = Image.open(bgpath)
        mask2 = self.transform_mask(im_bg) # mask out background
        mask2 = jt.array(mask2)
        mask2 = (mask2 >= 0.5).float() # foreground: 1, background: 0
        item_A_l['hair_A'] = (item_A/2+0.5) * mask.repeat(3,1,1) * mask2.repeat(3,1,1) * 2 - 1
        item_A_l['bg_A'] = (item_A/2+0.5) * (jt.ones(mask2.shape)-mask2).repeat(3,1,1) * 2 - 1

        return item_A, item_A_l['eyel_A'], item_A_l['eyer_A'], item_A_l['nose_A'], item_A_l['mouth_A'], item_A_l['hair_A'], item_A_l['bg_A'], mask, mask2, center, cmasks[0], cmasks[1], cmasks[2], cmasks[3]