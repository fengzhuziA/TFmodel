# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import argparse
from os.path import join as pjoin
import os
from xml.sax.saxutils import prepare_input_source
import torch
import torch.nn as nn
import numpy as np

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from os.path import join, dirname
import cv2
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.transformer import Transformer_Encoder
def normalize_reverse(x, centralize=True, normalize=True, val_range=255.0):
    if normalize:
        x = x * val_range
    if centralize:
        x = x + val_range / 2

    return x
def quantize( img):
    img = normalize_reverse(img)
    img = img.clamp(0, 255).round()
    return img

def normalize(x, centralize=False, normalize=False, val_range=255.0):
    if centralize:
        x = x - val_range / 2
    if normalize:
        x = x / val_range

    return x

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)

def prepare(para, x):
    rgb = 255.0
    if not para.uncentralized:
        x = x - rgb / 2
    if para.normalized:
        x = x / rgb
    # x = x - rgb / 2
    # x = x / rgb
    return x


def prepare_reverse(para, x):
    rgb = 255.0
    if para.normalized:
        x = x * rgb
    if not para.uncentralized:
        x = x + rgb / 2
    # x = x * rgb
    # x = x + rgb / 2

    return x

def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError

# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate,action='gelu'):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(act=action)

    def forward(self, x):
        out =self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out

# Middle network of residual dense blocks
class RDNet(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer))
        self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
        self.conv3x3 = conv3x3(in_channels, in_channels)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        return out

# DownSampling module
class RDB_DS(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer)
        self.down_sampling = conv5x5(in_channels, 2 * in_channels, stride=2)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out



# RDB-based RNN cell
class RDBCell(nn.Module):
    def __init__(self, para):
        super(RDBCell, self).__init__()
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.F_B0 = conv5x5(3, self.n_feats, stride=1)
        self.F_B1 = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=para.rd_layer)
        self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), num_layer=para.rd_layer)
        self.F_B3 = RDB_DS(in_channels=4 * self.n_feats,growthRate=int(self.n_feats * 3 / 2), num_layer=para.rd_layer)
        
        # self.F_R = RDNet(in_channels=8 * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
        #                  num_blocks=self.n_blocks)  # in: 80
        #

    def forward(self, x):
        #print('test_conv',self.F_B0.weight[0][0])
        F_B0 = self.F_B0(x)
        F_B1 = self.F_B1(F_B0)
        F_B2 = self.F_B2(F_B1)
        F_B3 = self.F_B3(F_B2)
        # F_final = self.F_R(F_B3)
   
     

        return F_B0,F_B1,F_B2,F_B3

# Reconstructor
class Reconstructor(nn.Module):
    def __init__(self, para):
        super(Reconstructor, self).__init__()
        self.para = para
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.n_feats = para.n_features
        self.model = nn.Sequential(
            nn.ConvTranspose2d((5 * self.n_feats) * (self.related_f), 2 * self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            conv5x5(self.n_feats, 3, stride=1)
        )

    def forward(self, x):
        return self.model(x)



def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis=True):
        super(Attention, self).__init__()
        self.vis = vis
        self.config = config
        self.query = Linear(config.in_channel, config.in_channel)
        self.key = Linear(config.in_channel, config.in_channel)
        self.value = Linear(config.in_channel, config.in_channel)

        self.out = Linear(self.config.in_channel, self.config.in_channel)
        self.attn_dropout = Dropout(config.attention_dropout)
        self.proj_dropout = Dropout(config.attention_dropout)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        b,n,h,w,c = x.shape
        x = x.view(b,n,-1,c)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.config.in_channel)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        b,n,hw,c=context_layer.shape
        context_layer =context_layer.view(b,n,self.config.hight,self.config.width,c)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.in_channel, config.in_channel*2)
        self.fc2 = Linear(config.in_channel*2, config.in_channel)
        self.act_fn = ACT2FN[config.act_fun]
        self.dropout = Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
       
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.attention_norm = LayerNorm(config.in_channel, eps=1e-6)
        self.ffn_norm = LayerNorm(config.in_channel, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis=True):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.in_channel, eps=1e-6)
        for _ in range(config.num_layer):
            layer = Block(config,vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

    
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, n_frames,height,width,d_model, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
       
        position = torch.arange(0, height*width).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 1).float() *
                             -(math.log(10000.0) / d_model))
        hw_pe= nn.Parameter(torch.sin(position * div_term).view(height,width,-1))
        hw_pe=hw_pe.unsqueeze(0).repeat(n_frames,1,1,1)
        #######
        
        pe = torch.zeros(n_frames, d_model)
        f_position = torch.arange(0, n_frames).unsqueeze(1).float()
        f_div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(f_position * f_div_term)
        pe[:, 1::2] = torch.cos(f_position * f_div_term)
        pe =nn.Parameter(pe.unsqueeze(1).unsqueeze(1))
        final_pe = pe+hw_pe
        self.final_pe = final_pe.unsqueeze(0)
        
        # self.register_buffer('pe', pe)

    def forward(self, x):
        # print('PE_test',self.final_pe)
        x = x + self.final_pe.to(x.device)
        return self.dropout(x)
    
class Transformer_Encoder(nn.Module):
    def __init__(self ,
                 hight,
                 width,
                 n_frames,
                 in_channel,
                 num_layer,
                 act_fun='gelu', ##"swish","gelu","relu"
                 attention_dropout=0.1,
                 dropout=0.1
                 ):
        super(Transformer_Encoder, self).__init__()
        # self.config = self.parser_opt()
        self.config = self.Args()
        self.config.hight = hight
        self.config.width = width
        self.config.in_channel = in_channel
        self.config.num_layer = num_layer
        self.config.act_fun=act_fun
        self.config.attention_dropout=attention_dropout
        self.config.dropout=dropout
        self.config.n_frames =n_frames
        
        self.position_embedding = PositionalEncoding(self.config.n_frames,self.config.hight,self.config.width,
                                                     self.config.in_channel,self.config.dropout)
        self.encoder = Encoder(self.config)
        
        # assert self.config.hidden_size % self.config.num_heads ==0

    
    def forward(self,input): ##input[b,n_frames,h,w,n_features]
        embedding_out = self.position_embedding(input)
        # b,n,h,w,f = embedding_out.shape
        # embedding_out = embedding_out.view(b,n,-1)
        
        encoder_out, attn_weights = self.encoder(embedding_out)
        return encoder_out, attn_weights
    class Args():
        def __init__(self):
            self.hight = None
            self.width = None
            self.in_channel = None
            self.num_layer = None
            self.act_fun = None
            self.attention_dropout = 0.1
            self.dropout = 0.1
            self.n_frames = None
    

# TFModel
class Model(nn.Module):
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        self.n_feats = para.n_features
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.ds_ratio = 4
        self.device = torch.device('cpu')
        self.cell = RDBCell(para)
        self.transformer_encoder = Transformer_Encoder(hight=32,
                                                       width=32,
                                                       n_frames=self.para.frames,
                                                       in_channel=8*self.n_feats,
                                                       num_layer=self.para.transformer_layer,
                                                       act_fun='gelu',
                                                       attention_dropout=0.1,
                                                       dropout=0.1
                                                       )
     

        self.Res_B3 = nn.ConvTranspose2d((8 * self.n_feats), 4 * self.n_feats, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.Res_RD3 = RDB(4 * self.n_feats, int(self.n_feats * 3 / 2), num_layer=para.rd_layer)

        self.Res_B2 = nn.ConvTranspose2d((4 * self.n_feats), 2 * self.n_feats, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.Res_RD2 = RDB(2 * self.n_feats, int(self.n_feats * 3 / 2), num_layer=para.rd_layer)
        self.Res_B1 = nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        self.Res_RD1 = RDB(self.n_feats, int(self.n_feats * 3 / 2), num_layer=para.rd_layer)
        self.Res_B0 = conv5x5(self.n_feats, 3, stride=1)
        
        
        # self.fusion = GSA(para)

    def forward(self, x, profile_flag=False):
        # profile_flag=False
        if profile_flag:
            return self.profile_forward(x)
        # x = prepare(self.para, x)
        outputs, hs = [], []
        batch_size, frames, channels, height, width = x.shape

        F_B0L, F_B1L, F_B2L,F_B3L =[],[],[],[]
        for i in range(frames):
            F_B0, F_B1, F_B2 ,F_B3= self.cell(x[:, i, :, :, :])
            F_B0L.append(F_B0)
            F_B1L.append(F_B1)
            F_B2L.append(F_B2)
            F_B3L.append(F_B3)
         
            encoder_out = F_B3.permute(0,2,3,1)
            # encoder_out = encoder_out.view(batch_size,-1,self.para.n_features*4)
            encoder_out = encoder_out.unsqueeze(1)
            hs.append(encoder_out)
        encoder_out = torch.cat(hs,1)
        trans_out,_ = self.transformer_encoder(encoder_out)

        trans_out = trans_out.permute(0,1,4,2,3)


        # for i in range(self.num_fb, frames - self.num_ff):
        for i in range(self.para.frames):
            B3 = self.Res_B3(trans_out[:,i,:,:,:] + F_B3L[i])
            B3RD = self.Res_RD3(B3)
            
            B2 = self.Res_B2(B3RD + F_B2L[i])
            B2RD = self.Res_RD2(B2)
            B1 = self.Res_B1(B2RD+ F_B1L[i])
            B1RD = self.Res_RD1(B1)
            out = self.Res_B0(B1RD +F_B0L[i])

            outputs.append(out.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1),_

class Get_Model(nn.Module):
    def __init__(self, para):
        super(Get_Model, self).__init__()
        self.para = para
        self.model =  Model(para)

    def __repr__(self):
        return self.model.__repr__() + '\n'

    def forward(self, x):
        return self.model(x)

def get_start_points(step,H,patch_size):

    H_steps = 0
    while True:
        if H_steps*step+ patch_size>H:
            break
        else:
            H_steps+=1
    H_points=[]
    for i in range(H_steps):
        H_points.append(i*step)
        if i ==H_steps-1:
            H_points.append(H - patch_size)
    return H_points
def gene_whole_img(para, inputs, model,gap=10):
    ####
    patch_size = para.patch_size
    step = patch_size - gap
    b, f, c, H, W = inputs.shape
    
  

    H_points = get_start_points(step=step,H=H,patch_size=patch_size)
    W_points = get_start_points(step=step,H=W,patch_size=patch_size)
    

    outputs = torch.zeros(b, f, c, H, W).to(para.device)
    weights = torch.zeros(b, f, c, H, W).to(para.device)

    for s_h in H_points:
        for s_w in W_points:
            patch_inputs = inputs[:, :, :, s_h:s_h+patch_size, s_w:s_w + patch_size]
            patch_inputs = patch_inputs
            patch_out, attention_weights= model(patch_inputs)
            outputs[:, :, :, s_h:s_h + patch_size, s_w:s_w + patch_size] += patch_out
            weights[:, :, :, s_h:s_h + patch_size, s_w:s_w + patch_size] += 1.0
    outputs = outputs / weights
    return outputs
def load_check(para):# model_path是权重文件的路径
    model_path = para.resume_file
    ckpt= torch.load(model_path, map_location='cpu')['state_dict']
    our_model = Model(para) # 初始化网络模型
    new_ckpt =copy.copy(ckpt)
    for name in ckpt.keys():
        new_name = name.split('module.')[1]
        value = ckpt[name]
        new_ckpt.pop(name)
        new_ckpt[new_name]=value
    our_model.load_state_dict(new_ckpt) # 加载权重文件
    return our_model

def split_video(video_file,out_dir):
    
    cap=cv2.VideoCapture(video_file)

    if cap.isOpened():
        ret,frame=cap.read()
    else:
        ret = False

    n=0
    i=0
    timeF = 1
    path=out_dir+'/{}'
    while ret:
        n = n + 1
        ret,frame=cap.read()
        if (n%timeF == 0) :
            i = i+1
            filename=str(i)+".jpg"
            if ret == True:
                cv2.imwrite(path.format(filename),frame)

    cap.release()

def get_img_files(img_dir):
    files = os.listdir(img_dir)

    files.sort(key=lambda x: int(x.split('.')[0])) ## 使用切片将图片名称单独切开
    img_files=[]
    for path in files:
        full_path = os.path.join(para.img_dir, path)
        img_files.append(full_path)
    return img_files

def pic2video(para):
    img = cv2.imread(para.deblurimg_dir+'/2.jpg')
    imginfo = img.shape
    size = (imginfo[1],imginfo[0])
    print(size)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWrite = cv2.VideoWriter(para.output_video,fourcc,10,size) 
    deblur_files = get_img_files(para.deblurimg_dir)

    for filename in deblur_files:
        
        img = cv2.imread(filename,1)
        videoWrite.write(img)  

    videoWrite.release()
    print('img2video is end !')
    
def prepare_imgs_from_video(para):
    split_video(para.input_video,para.img_dir)

    img_files = get_img_files(para.img_dir)

    f_length = len(img_files)

    frame_index = list(range(0, f_length, para.frames))
    if f_length%para.frames !=0:
        frame_index[-1] = f_length-para.frames
    return img_files,frame_index
    

def write_to_dir(para,img_files,output_seq,index):
    img_names=img_files[index:index+para.frames]
    deblur_img_names=[]
    for oldname in img_names:
        newname = oldname.replace(para.img_dir,para.deblurimg_dir)
        deblur_img_names.append(newname)
    for idx in range(para.frames):
        deblur_img = output_seq[:,idx].squeeze(0) #
        deblur_img = normalize_reverse(deblur_img, centralize=True, normalize=True)
        deblur_img = deblur_img.detach().cpu().numpy().transpose((1, 2, 0))
        deblur_img = np.clip(deblur_img, 0, 255).astype(np.uint8)
        deblur_img_path = deblur_img_names[idx]
        cv2.imwrite(deblur_img_path, deblur_img)


def run(para,our_model):
    img_files,frame_index = prepare_imgs_from_video(para)

    for index in frame_index:
        
        input_seq=[]
        for seq in range(para.frames):
            input_seq.append( cv2.imread(img_files[index+seq]).transpose((2, 0, 1))[np.newaxis, :])
        input_seq = np.concatenate(input_seq)[np.newaxis, :]
        input_seq =  normalize(torch.from_numpy(input_seq).float().to(para.device), centralize=True,
                                        normalize=True)
        output_seq = gene_whole_img(para=para,inputs=input_seq,model = our_model)
        
        write_to_dir(para,img_files,output_seq,index) #
    pic2video(para)

class Parameter:
    def __init__(self):
        self.args = self.extract_args()

    def extract_args(self):
        self.parser = argparse.ArgumentParser(description=' Video Deblurring')

        # need to be set
        self.parser.add_argument('--resume_file', type=str, default=".Ckpt/F5R4T9H6.pth.tar")
        self.parser.add_argument('--device', type=str, default='cpu', help='')
        self.parser.add_argument('--input_video', type=str, default='./demo.mp4', help='')
        self.parser.add_argument('--output_video', type=str, default='./deblur.mp4', help='')
        self.parser.add_argument('--img_dir', type=str, default='./imgs', help='')
        self.parser.add_argument('--deblurimg_dir', type=str, default='./deblurimgs', help='')
      
        #parameters
        self.parser.add_argument('--transformer_layer', type=int, default=9, help='num layer')
        self.parser.add_argument('--rd_layer', type=int, default=3, help='num layer')
        self.parser.add_argument('--num_gpus', type=int, default=1, help='# of GPUs to use')
        self.parser.add_argument('--frames', type=int, default=5, help='# of frames of subsequence')
        self.parser.add_argument('--patch_size', type=int, nargs='*', default=256)
        self.parser.add_argument('--n_features', type=int, default=16*6, help='base # of channels for Conv')
        self.parser.add_argument('--n_blocks', type=int, default=0, help='# of blocks in middle part of the model')
        self.parser.add_argument('--future_frames', type=int, default=0, help='use # of future frames')
        self.parser.add_argument('--past_frames', type=int, default=0, help='use # of past frames')
        self.parser.add_argument('--uncentralized', action='store_true', help='do not subtract the value of mean')
        self.parser.add_argument('--normalized', action='store_true', help='divide the range of rgb (255)')
        args, _ = self.parser.parse_known_args()

        return args



para = Parameter().args

our_model = load_check(para).to(para.device)
run(para,our_model)

    




    

  




