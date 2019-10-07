import numpy as np
import cv2
import math
from torchvision.transforms.functional import adjust_brightness, adjust_contrast
from PIL import Image


def random_rotate(img, pts, rot_ang_min, rot_ang_max):
    # 图像和特征点同时旋转随机角度
    h, w = img.shape[0], img.shape[1]
    degree = np.random.uniform(rot_ang_min,rot_ang_max)
    R = cv2.getRotationMatrix2D((w/2,h/2), degree, 1)
    img = cv2.warpAffine(img,R,(w, h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)

    pts=np.concatenate([pts,np.ones([pts.shape[0],1])],1) # n,3
    last_row=np.asarray([[0,0,1]],np.float32)
    pts=np.matmul(pts, np.concatenate([R, last_row], 0).transpose())

    return img, pts[:, :2]


def random_rotate_img(img, rot_ang_min, rot_ang_max):
    # 图像旋转随机角度
    img = np.array(img)
    h, w = img.shape[0], img.shape[1]
    degree = np.random.uniform(rot_ang_min, rot_ang_max)
    R = cv2.getRotationMatrix2D((w/2, h/2), degree, 1)
    img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return Image.fromarray(img)


def crop_or_pad_to_fixed_size(img, pts=None, ratio=1.0):
    # 对图像和特征点同时进行随机缩小（随机填充到原大小）或放大（同时随机裁剪到原大小）
    h,w=img.shape[0],img.shape[1]
    th,tw=int(math.ceil(h*ratio)),int(math.ceil(w*ratio))
    img=cv2.resize(img,(tw,th),interpolation=cv2.INTER_LINEAR)
    if pts is not None:
        pts*=ratio

    if ratio>1.0:
        # crop
        hbeg,wbeg=np.random.randint(0,th-h),np.random.randint(0,tw-w)
        result_img=img[hbeg:hbeg+h,wbeg:wbeg+w]
        if pts is not None:
            pts[:, 0] -= wbeg
            pts[:, 1] -= hbeg
    else:
        # padding
        if len(img.shape)==2:
            result_img=np.zeros([h,w],img.dtype)
        else:
            result_img = np.zeros([h, w, img.shape[2]], img.dtype)
        hbeg,wbeg=(h-th)//2,(w-tw)//2
        result_img[hbeg:hbeg+th,wbeg:wbeg+tw]=img
        if pts is not None:
            pts[:, 0] += wbeg
            pts[:, 1] += hbeg

    if pts is not None:
        return result_img, pts
    else:
        return result_img


def add_noise(image):
    # 随机增加高斯噪声、运动模糊
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.75:
        row,col,ch= image.shape
        mean = 0
        var = np.random.rand(1) * 0.3 * 256
        sigma = var**0.5
        gauss = sigma * np.random.randn(row,col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype(np.uint8)
    else:
        # motion blur
        sizes = [3, 5, 7, 9]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)
    return noisy


def gaussian_blur(img,blur_range=[3,5,7,9,11]):
    sigma=np.random.choice(blur_range,1)
    return cv2.GaussianBlur(img,(sigma,sigma),0)


def jpeg_compress(img,quality_low=15,quality_high=75):
    quality=np.random.randint(quality_low,quality_high)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    img=cv2.imdecode(encimg,1)
    return img


def additive_gaussian_noise(image, stddev_range=(5, 95)):
    # 加性高斯噪声（并非模糊）
    stddev = np.random.uniform(*stddev_range)
    noise = np.random.normal(0.0 , stddev, image.shape) #高斯分布
    noisy_image = image.astype(np.float32)+noise
    noisy_image=np.clip(noisy_image,0,255)
    return noisy_image.astype(np.uint8)


def additive_speckle_noise(image, prob_range=(0.0, 0.005)):
    # 加性椒盐噪声
    prob = np.random.uniform(*prob_range)
    sample = np.random.rand(*image.shape)
    noisy_image = image.astype(np.float32).copy()
    noisy_image[sample<prob] = 0.0
    noisy_image[sample>=(1-prob)] = 255
    return noisy_image.astype(np.uint8)


def random_brightness(image, max_change=0.3):
    return np.asarray(adjust_brightness(Image.fromarray(image),max_change))


def random_contrast(image, max_change=0.5):
    return np.asarray(adjust_contrast(Image.fromarray(image),max_change))


def additive_shade(img, nb_ellipses=20, transparency_range=(-0.5, 0.8),
                   kernel_size_range=(250, 350)):
    # 对图像增加加性随机椭圆阴影
    img=img.astype(np.float32)
    min_dim = min(img.shape[:2]) / 4
    mask = np.zeros(img.shape[:2], np.uint8)
    for i in range(nb_ellipses):
        ax = int(max(np.random.rand() * min_dim, min_dim / 5))
        ay = int(max(np.random.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
        y = np.random.randint(max_rad, img.shape[0] - max_rad)
        angle = np.random.rand() * 90
        cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

    transparency = np.random.uniform(*transparency_range)
    kernel_size = np.random.randint(*kernel_size_range)
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1
    mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    shaded = img * (1 - transparency * mask[..., np.newaxis]/255.)
    return np.clip(shaded, 0, 255).astype(np.uint8)


def motion_blur(img, max_kernel_size=10):
    # Either vertial, hozirontal or diagonal blur
    img = np.array(img).astype(np.float32)
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
    center = int((ksize-1)/2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    img = cv2.filter2D(img, -1, kernel)
    img = np.clip(img, 0, 255)
    return Image.fromarray(img.astype(np.uint8))


def resize_blur(img, max_ratio=0.15):
    h, w, _ = img.shape
    ratio_w, ratio_h = np.random.uniform(max_ratio, 1.0, 2)
    return cv2.resize(cv2.resize(img, (int(w*ratio_w), int(h*ratio_h)), interpolation=cv2.INTER_LINEAR),
                      (w, h), interpolation=cv2.INTER_LINEAR)