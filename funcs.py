import os
import cv2
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.optim import LBFGS
from torchvision.models import vgg19, VGG19_Weights
from tqdm import tqdm

from ESRGAN import RRDBNet_arch as arch

def apply_colormap(original_img, colormap_img):
    '''
    A function to apply colormap of colormap image to original image
    original_img - np.array, original image in BGR format
    colormap_img - np.array, colormap image in BGR format
    '''
    original_lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
    colormap_lab = cv2.cvtColor(colormap_img, cv2.COLOR_BGR2LAB)

    original_ch = np.array(cv2.split(original_lab))
    colormap_ch = np.array(cv2.split(colormap_lab))

    result = original_ch - np.mean(original_ch, axis=(1, 2), keepdims=True)
    result *= np.std(colormap_ch, axis=(1, 2), keepdims=True)
    result /= np.std(original_ch, axis=(1, 2), keepdims=True)
    result += np.mean(colormap_ch, axis=(1, 2), keepdims=True)

    result = np.clip(result, 0, 255)
    result = cv2.merge(result)
    result = np.uint8(result)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def ESRGAN_upscaler(image, model_path, device='cpu'):
    '''
    4x Upscaling images with ESRGAN
    image - image to upscale
    model_path - path to model's weights
    '''

    device = torch.device(device)

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    img = image.copy()
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    output = np.uint8(output)
        
    return output

class ImgPreparer():
    def __init__(self, original_img, style_img, device='cpu'):
        '''
        original_img - image to be stylized
        style_img - source of style for original_img
        '''
        self.original = original_img
        self.style = style_img
        self.device = torch.device(device)
    
    def downscale(
            self, downscale_limit, 
            do_blur=True, blur_params={'ksize':(3, 3), 'sigmaX':1.5},
            apply_style_cmap=True
        ):
        '''
        Reduces the size of the image and prepares laplacians for further upscaling

        downscale_limit - maximum number of pixels in downscaled picture. 
        do_blur - if True, gaussian blur is applied before downcaling
        blur_params - parameters of cv2.GaussianBlur
        apply_style_cmap - if True, colormap of style image is applied before downscaling
        '''
        
        downed = self.original.copy()
        if do_blur:
            downed = cv2.GaussianBlur(downed, **blur_params)
        if apply_style_cmap:
            downed = apply_colormap(downed, self.style)
        gaussian = [downed]
        while downed.shape[0] * downed.shape[1] > downscale_limit:
            downed = cv2.pyrDown(downed)
            gaussian.append(downed)
        self.gaussian = gaussian

        laplacian = [gaussian[-1]]
        for i in range(len(gaussian) - 1, 0, -1):
            h, w, _ = gaussian[i-1].shape
            up = cv2.pyrUp(gaussian[i], dstsize=(w, h))
            diff = cv2.subtract(gaussian[i-1], up)
            laplacian.append(diff)
        self.laplacian = laplacian

    def upscale(
            self, img, 
            resize_only=False, use_ESRGAN=None, 
            ESRGAN_weights_path='ESRGAN/models/RRDB_ESRGAN_x4.pth'
        ):
        '''
        Returns upscaled version of the image

        img - stylized original image to upscale. 
        resize_only - if True, the image is only returned to it's original size  
        use_ESRGAN - str or None.
        - If None - ESRGAN is not used
        - If 'first' - ESRGAN is used instead of first two pyramid upscales  
        - If 'last' - ESRGAN is used instead of last two pyramid upscales  
        - If  'all' - ESRGAN is used instead of all pyramid upscales   
        
        Usin ESRGAN takes more time than upscaling with laplacian pyramids
        '''
        if resize_only:
            h, w, _ = self.laplacian[-1].shape
            resized = cv2.resize(img, (w, h))
            self.upscaled = upscaled = [img, resized]
            return resized

        
        current = img
        upscaled = [current]
        laplacian = self.laplacian
        if use_ESRGAN == 'all':
            # We one use of ESRGAN (x4) equals two uses of pyrUp (x2)
            for i in range(0, len(laplacian), 2):
                current = ESRGAN_upscaler(current, ESRGAN_weights_path)
                if i + 2 < len(laplacian):
                    h, w, _ = laplacian[i + 2].shape
                else:
                    h, w, _ = laplacian[-1].shape
                current = cv2.resize(current, (w, h))
                upscaled.append(current)
        else:
            lap_start = 1
            lap_end = len(laplacian)
            if use_ESRGAN == 'first':
                current = ESRGAN_upscaler(current, ESRGAN_weights_path)
                if len(laplacian) > 2:
                    h, w, _ = laplacian[2].shape 
                else:
                    h, w, _ = laplacian[-1].shape 
                current = cv2.resize(current, (w, h))
                upscaled.append(current)
                lap_start += 2
            elif use_ESRGAN == 'last':
                lap_end -= 2

            for i in range(lap_start, lap_end):
                h, w, _ = laplacian[i].shape
                current = cv2.pyrUp(current, dstsize=(w, h))
                current = cv2.add(current, laplacian[i])
                upscaled.append(current)
            
            if use_ESRGAN == 'last':
                current = ESRGAN_upscaler(current, ESRGAN_weights_path)
                h, w, _ = laplacian[-1].shape 
                current = cv2.resize(current, (w, h))
                upscaled.append(current)

        self.upscaled = upscaled
        return current

    def prepare_imgs(
            self, start_from_gauss=False, 
            apply_style_cmap=True, downscale_limit=35000):
        '''
        Function prepares original, style and input (trainable) images to style transfer with PyTorch.

        start_from_gauss - if True, the input image is constructed from gaussian noise; 
        otherwise the original image is copied 
        apply_style_cmap - if True, colormap of style image is to the input image.
        downscale_limit - maximum number of pixels in downscaled picture. 
        Default is 35000, which is less than in 256x256 image, but more than in 128x128 image
        '''
        self.downscale(
            downscale_limit=downscale_limit,
            apply_style_cmap=apply_style_cmap
        )
        original = self.original.copy()
        style = self.style.copy()
        
        h, w, _ = self.gaussian[-1].shape

        if start_from_gauss:
            input = np.random.normal(size=original.shape)
            input = np.uint8(input*255)
        else:
            input = original.copy()
        if apply_style_cmap:
            input = apply_colormap(input, self.style)

        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((h, w))]
        )
        original = transformer(original)  
        original = original[None, ...]
        original = original.to(self.device, torch.float)
        
        style = transformer(style)
        style = style[None, ...]
        style = style.to(self.device, torch.float)

        input = transformer(input)
        input = input[None, ...]
        input = input.to(self.device, torch.float)

        
        return original, style, input

    def restore_img(self, img_, needs_upscale=False, upscale_params={}):
        '''
        Restores images from torch-suitable format to BGR numpy array
        '''
        if isinstance(img_, torch.Tensor):
            img = img_.clone()
            img = img.detach().cpu().numpy()
        else:
            img = img_.copy()
        img = img[0].swapaxes(0, 2).swapaxes(0, 1)
        img *= 255
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if needs_upscale:
            self.upscale(img, **upscale_params)
            img = self.upscaled[-1]
        return img

    def remove_color_noise(self, img, params={}):
        '''
        Removes color noise from image. Basically a wrapper for cv2.fastNlMeansDenoisingColored.
        
        img - image to remove color noise from
        h - parameter deciding filter strength. Higher h value removes noise better, but removes details of image also.
        hColor - same as h, but for color images only
        templateWindowSize - should be odd. Default: 5
        searchWindowSize - should be odd. Default: 10
        '''
        params_ = {
            'dst': None, 
            'h': 5, 
            'hColor': 2, 
            'templateWindowSize': 5, 
            'searchWindowSize': 10
        }       
        for key in params:
            if key in params_:
                params_[key] = params[key]
        return cv2.fastNlMeansDenoisingColored(img, **params_)

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    
def gram_matrix(input):
    a, b, c, d = input.size()  
    features = input.reshape(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        self.targets = []
        for target_feature in target_features:
            target = gram_matrix(target_feature).detach()
            self.targets.append(target)
        self.targets = torch.stack(self.targets)

    def forward(self, input):
        G = gram_matrix(input)   
        self.loss = F.mse_loss(G.unsqueeze(0), self.targets)          
        return input
    
def load_vgg19():    
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    return cnn, cnn_normalization_mean, cnn_normalization_std

def get_name_idx_dicts(cnn):
    conv_id = 1 # number of conv layer in block
    relu_id = 1 # number of relu layer in block
    block_id = 1 # number of the block 
    # name example - 'conv_23' - third conv layer in the second block
    name2idx = dict()
    modules = dict(cnn.named_modules())
    for key, module in modules.items():
        if key == '':
            continue
        if isinstance(module, nn.Conv2d):
            name2idx[f'conv_{block_id}{conv_id}'] = key
            conv_id += 1
        elif isinstance(module, nn.ReLU):
            name2idx[f'relu_{block_id}{relu_id}'] = key
            relu_id += 1
        elif isinstance(module, nn.MaxPool2d):
            name2idx[f'pool_{block_id}'] = key
            block_id += 1
            conv_id = 1
            relu_id = 1
    idx2name = {v: k for k, v in name2idx.items()}
    return name2idx, idx2name

# additionaly, we normalize images the way they were normalized while Vgg19 was trained.
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().reshape(-1, 1, 1)
        self.std = std.clone().reshape(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def get_style_model_and_losses(
        cnn, normalization_mean, normalization_std,
        style_img, content_img,
        content_layers=['conv_22'],
        style_layers=['conv_11', 'conv_12', 'conv_21', 'conv_22', 'conv_31']
    ):
    
    name2idx, idx2name = get_name_idx_dicts(cnn)

    content_set = set(content_layers)
    style_set = set(style_layers)

    content_losses = []
    style_losses = []
    
    normalization = Normalization(normalization_mean, normalization_std)
    model = nn.Sequential(normalization)
    for idx, module in dict(cnn.named_modules()).items():
        if idx == '':
            continue
        if len(style_set) + len(content_set) < 1:
            break
        name = idx2name[idx]
        if 'relu' in name:
            model.add_module(name, nn.ReLU(inplace=False))
        else:
            model.add_module(name, module)
        if name in content_set:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{name}', content_loss)
            content_losses.append(content_loss)
            content_set.remove(name)
        
        if name in style_layers: 
            target_feature = model(style_img).detach()
            style_loss = StyleLoss([target_feature])
            model.add_module(f'style_loss_{name}', style_loss)
            style_losses.append(style_loss)
            style_set.remove(name)

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    optimizer = LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1e6, content_weight=1):

    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    imgs=[]
    run = [0]
    pbar = tqdm(total=num_steps, position=0, leave=True)
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            
            # get losses
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            # progressbar update
            run[0] += 1
            pbar.update(1)
            pbar.set_description(
                f'Style Loss : {style_score.item():.4}, '
                f' Content Loss: {content_score.item():.4}'
            )
            # saving intermediate results
            imgs.append(input_img.clone().clamp_(0, 1).detach().cpu().numpy())
            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    imgs.append(input_img.clone().detach().cpu().numpy())
    
    return input_img, imgs

