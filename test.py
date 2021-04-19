import argparse
import os
import numpy as np
import math
import datetime
import time

from models import *
from datasets import *
from utils import *

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, default="./samples/A_img/example", help="the folder of input photos")
parser.add_argument("--lm_folder", type=str, default="./samples/A_landmark/example", help="the folder of input landmarks")
parser.add_argument("--mask_folder", type=str, default="./samples/A_mask_bg/example", help="the folder of foreground masks")
parser.add_argument("--cmask_folder", type=str, default="./samples/A_cmask_{}/example", help="the folder of compact masks")
parser.add_argument("--model_name", type=str, default="apdrawinggan++_author", help="the load folder of model")
parser.add_argument("--which_epoch", type=int, default=150, help="number of epoch to load")
parser.add_argument("--dataset_name", type=str, default="portrait_drawing", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--in_channels", type=int, default=3, help="number of input channels")
parser.add_argument("--out_channels", type=int, default=1, help="number of output channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--extra_channel", type=int, default=3, help="extra channel for style feature")
opt = parser.parse_args()
print(opt)

# Create save directories
data_subdir = opt.input_folder.split('/')[-1]
save_folder = "results/%s/%s_%d/%s" % (opt.dataset_name, opt.model_name, opt.which_epoch, data_subdir)
os.makedirs(save_folder, exist_ok=True)

input_shape = (opt.in_channels, opt.img_height, opt.img_width)
output_shape = (opt.out_channels, opt.img_height, opt.img_width)

# Initialize generator
G_global = GeneratorResNet(in_channels=opt.in_channels, out_channels=opt.out_channels)
G_l_eyel = GeneratorResNet(in_channels=opt.in_channels, out_channels=opt.out_channels, num_res_blocks=3)
G_l_eyer = GeneratorResNet(in_channels=opt.in_channels, out_channels=opt.out_channels, num_res_blocks=3)
G_l_nose = GeneratorResNet(in_channels=opt.in_channels, out_channels=opt.out_channels, num_res_blocks=3)
G_l_mouth = GeneratorResNet(in_channels=opt.in_channels, out_channels=opt.out_channels, num_res_blocks=3)
G_l_hair = GeneratorResStyle2Net(in_channels=opt.in_channels, out_channels=opt.out_channels, num_res_blocks=6)
G_l_bg = GeneratorResNet(in_channels=opt.in_channels, out_channels=opt.out_channels, num_res_blocks=6)
G_combine = Combiner(in_channels=2*opt.out_channels, out_channels=opt.out_channels)

ratio = opt.img_height // 256
rhs = np.array([EYE_H,EYE_H,NOSE_H,MOUTH_H]) * ratio
rws = np.array([EYE_W,EYE_W,NOSE_W,MOUTH_W]) * ratio
ae_eyel = AutoEncoderWithFC(input_nc=1, output_nc=1, h=rhs[0], w=rws[0])
ae_eyer = AutoEncoderWithFC(input_nc=1, output_nc=1, h=rhs[1], w=rws[1])
ae_nose = AutoEncoderWithFC(input_nc=1, output_nc=1, h=rhs[2], w=rws[2])
ae_moblack = AutoEncoderWithFC(input_nc=1, output_nc=1, h=rhs[3], w=rws[3])
ae_mowhite = AutoEncoderWithFC(input_nc=1, output_nc=1, h=rhs[3], w=rws[3])
CLm = Classifier(input_nc=3, classes=2, num_downs=3, h=rhs[3], w=rws[3])
CLh = Classifier(input_nc=3, classes=3, num_downs=6, h=opt.img_height, w=opt.img_width)
ae_nose.load('checkpoints/auxiliaryeye2o/latest_net_AE.pth')
ae_eyel.load('checkpoints/auxiliaryeye2o/latest_net_AEel.pth')
ae_eyer.load('checkpoints/auxiliaryeye2o/latest_net_AEer.pth')
ae_moblack.load('checkpoints/auxiliaryeye2o/latest_net_AEmoblack.pth')
ae_mowhite.load('checkpoints/auxiliaryeye2o/latest_net_AEmowhite.pth')
CLm.load('checkpoints/auxiliaryeye2o/latest_net_CLm.pth')
CLh.load('checkpoints/auxiliaryeye2o/latest_net_CLh.pth')
print('loading auxiliary models done')

# Load weight
model_path = os.path.join("checkpoints", opt.model_name, "{}_net_gen.pth".format(opt.which_epoch))
if os.path.exists(model_path):
    state_dict = jt.safeunpickle(model_path)
    G_global.load_state_dict(state_dict['G'])
    G_l_eyel.load_state_dict(state_dict['GLEyel'])
    G_l_eyer.load_state_dict(state_dict['GLEyer'])
    G_l_nose.load_state_dict(state_dict['GLNose'])
    G_l_mouth.load_state_dict(state_dict['GLMouth'])
    G_l_hair.load_state_dict(state_dict['GLHair'])
    G_l_bg.load_state_dict(state_dict['GLBG'])
    G_combine.load_state_dict(state_dict['GCombine'])
else:
    G_global.load(os.path.join("checkpoints", opt.model_name, "{}_net_G_global.pkl".format(opt.which_epoch)))
    G_l_eyel.load(os.path.join("checkpoints", opt.model_name, "{}_net_G_l_eyel.pkl".format(opt.which_epoch)))
    G_l_eyer.load(os.path.join("checkpoints", opt.model_name, "{}_net_G_l_eyer.pkl".format(opt.which_epoch)))
    G_l_nose.load(os.path.join("checkpoints", opt.model_name, "{}_net_G_l_nose.pkl".format(opt.which_epoch)))
    G_l_mouth.load(os.path.join("checkpoints", opt.model_name, "{}_net_G_l_mouth.pkl".format(opt.which_epoch)))
    G_l_hair.load(os.path.join("checkpoints", opt.model_name, "{}_net_G_l_hair.pkl".format(opt.which_epoch)))
    G_l_bg.load(os.path.join("checkpoints", opt.model_name, "{}_net_G_l_bg.pkl".format(opt.which_epoch)))
    G_combine.load(os.path.join("checkpoints", opt.model_name, "{}_net_G_combine.pkl".format(opt.which_epoch)))

# Test data loader
test_dataloader = TestDataset(opt.input_folder, opt.lm_folder, opt.mask_folder, opt.cmask_folder, mode="test", load_h=opt.img_height, load_w=opt.img_width).set_attrs(batch_size=1, shuffle=False, num_workers=1)
import cv2
def save_single_image(img, path):
    N,C,W,H = img.shape
    img = img[0]
    min_ = -1
    max_ = 1
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = img[:,:,::-1]
    cv2.imwrite(path,img)

# ----------
#  Testing
# ----------

prev_time = time.time()
for i, batches in enumerate(test_dataloader):

    real_A = batches[0]
    real_A_eyel = batches[1]
    real_A_eyer = batches[2]
    real_A_nose = batches[3]
    real_A_mouth = batches[4]
    real_A_hair = batches[5]
    real_A_bg = batches[6]
    mask = batches[7]
    mask2 = batches[8]
    center = batches[9]
    cmaskel = batches[10]
    cmasker = batches[11]
    cmaskno = batches[12]
    cmaskmo = batches[13]

    maskh = mask*mask2
    maskb = inverse_mask(mask2)

    fake_B0 = G_global(real_A)

    # EYES, NOSE, MOUTH
    fake_B_eyel1 = G_l_eyel(real_A_eyel)
    fake_B_eyel2 = ae_eyel(fake_B_eyel1)
    fake_B_eyel = add_with_mask(fake_B_eyel2, fake_B_eyel1, cmaskel)
    fake_B_eyer1 = G_l_eyer(real_A_eyer)
    fake_B_eyer2 = ae_eyer(fake_B_eyer1)
    fake_B_eyer = add_with_mask(fake_B_eyer2, fake_B_eyer1, cmasker)
    fake_B_nose1 = G_l_nose(real_A_nose)
    fake_B_nose2 = ae_nose(fake_B_nose1)
    fake_B_nose = add_with_mask(fake_B_nose2, fake_B_nose1, cmaskno)
    fake_B_mouth1 = G_l_mouth(real_A_mouth)
    outputs1 = CLm(real_A_mouth)
    pred = jt.argmax(outputs1, dim=1)[0]
    fake_B_mouth2w = ae_mowhite(fake_B_mouth1)
    fake_B_mouth2b = ae_moblack(fake_B_mouth1)
    fake_B_mouth2s = jt.contrib.concat((fake_B_mouth2w,fake_B_mouth2b), 1)
    bs,c,h,w = fake_B_mouth2s.shape
    index = pred + jt.arange(bs) * c
    fake_B_mouth2 = fake_B_mouth2s.reshape([-1,h,w])[index].reshape([bs,1,h,w])
    fake_B_mouth = add_with_mask(fake_B_mouth2, fake_B_mouth1, cmaskmo)

    # HAIR & BG
    outputs2 = CLh(real_A_hair)
    onehot2 = getonehot(outputs2, 3, bs)
    fake_B_hair = G_l_hair(real_A_hair, onehot2)
    fake_B_bg = G_l_bg(real_A_bg)

    # PARTCOMBINE
    fake_B1 = partCombiner2_bg(center, fake_B_eyel, fake_B_eyer, fake_B_nose, fake_B_mouth, fake_B_hair, fake_B_bg, maskh, maskb, comb_op=1, load_h=opt.img_height, load_w=opt.img_width)
    
    # FUSION NET
    fake_B = G_combine(jt.contrib.concat((fake_B0, fake_B1), 1))

    save_single_image(real_A.numpy(), "%s/%d_real.png" % (save_folder, i))
    save_single_image(fake_B.numpy(), "%s/%d_fake.png" % (save_folder, i))
print("Test time: %.2f" % (time.time() - prev_time))