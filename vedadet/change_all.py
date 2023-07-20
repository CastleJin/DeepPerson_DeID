import cv2
import numpy as np
from torchvision.utils import save_image
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt
from vedacore.image import imread, imwrite
import glob
import os
import torch
import modifier
import argparse
from copy import deepcopy
from skimage.segmentation import slic
from skimage.measure import regionprops
from tqdm import tqdm
import torchvision.transforms as transforms
import ugnet_pixd
enc = transforms.Compose(
    [
        transforms.Resize([256, 128]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
inverse = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        transforms.Resize([128, 64]),
    ]
)
tt = transforms.Resize([128, 64])

def black(img):
    out = deepcopy(img)
    _, _, c = out.shape
    gray = np.mean(img, axis=2)
    for i in range(c):
        out[:, :, i] = gray
    return out


def L_blur(img, f1):
    out = deepcopy(img)
    h, w, c = out.shape
    for i in range(c):
        roi = cv2.resize(out[:, :, i], (w//f1, h//f1))
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
        out[:, :, i] = roi
    return out


def G_blur(img, sigma=7):  # 3
    out = deepcopy(img)
    out = cv2.GaussianBlur(out, (0, 0), sigma)
    return out


def Blank(img):
    out = deepcopy(img)
    out = np.ones_like(out) * 255
    return out


def Noise(img, sig=45):
    out = deepcopy(img)
    gauss = np.random.normal(0, sig, (out.shape[0], out.shape[1]))
    noise = np.zeros_like(out)
    for i in range(3):
        noise[:, :, i] = out[:, :, i] + gauss
    return noise


def Super(img, segment=50):
    out = deepcopy(img)

    def paint_region_with_avg_intensity(rp, mi, channel):
        for i in range(rp.shape[0]):
            out[rp[i][0]][rp[i][1]][channel] = mi

    segments = slic(
        out,
        n_segments=segment,
        compactness=10,
        channel_axis=2,
        enforce_connectivity=True,
        convert2lab=True,
    )

    for i in range(3):
        regions = regionprops(segments, intensity_image=out[:, :, i])
        for r in regions:
            paint_region_with_avg_intensity(r.coords, int(r.mean_intensity), i)
    return out


def Edge(img):
    out = deepcopy(img)
    gray = deepcopy(out)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    ed = cv2.Canny(gray, 30, 70)
    for i in range(3):
        out[:, :, i] = ed
    return out


def Deep(img):
    a = enc(img).cuda()
    out = modelG(a)
    return out


def denormalize(image):
    image = image * 255.0

    # IMAGENET 형식으로 normalize 된 경우
    IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array(
        [0.229, 0.224, 0.225]
    )
    image = np.clip(255.0 * (image * IMAGENET_STD + IMAGENET_MEAN), 0, 255)
    return image


def invert(img):
    result = inverse(img)
    return result


def Filter(cropped, args):
    if args.type == "mosiac":
        return L_blur(cropped, args.para)
    elif args.type == "blur":
        return G_blur(cropped, args.para)
    elif args.type == "mask":
        return Blank(cropped)
    elif args.type == "noise":
        return Noise(cropped, args.para)
    elif args.type == "super":
        return Super(cropped, args.para)
    elif args.type == "edge":
        return Edge(cropped)
    elif args.type == "gray":
        return black(cropped)
    else:
        print("Error. Check the current filter_state")


def set(name, args):
    i = "/" + args.type + name + "/market1501"
    j = args.out_path + i
    if not os.path.isdir(j):
        os.mkdir(args.out_path + "/" + args.type + name)
        os.mkdir(j)
        os.system("cp -r " + args.root + "/query" + " " + j + "/query")
        os.mkdir(j + "/bounding_box_test")
        os.system(
            "cp -r "
            + args.root
            + "/bounding_box_train"
            + " "
            + j
            + "/bounding_box_train"
        )
        os.system("cp -r " + args.root + "/gt_query" + " " + j + "/gt_query")
        os.system("cp -r " + args.root + "/gt_bbox" + " " + j + "/gt_bbox")


def crop(img, x, y, w, h):
    cropped_img = img[y : y + h, x : x + w]
    return cropped_img


def replace(img, c, x, y, w, h):
    img[y : y + h, x : x + w] = c
    return img


def renew(boxes):
    le = []
    is_it = -1
    for i in boxes:
        le.append(len(i))
    for num, item in enumerate(le):
        if item == 1:
            is_it = num

    if is_it > 0:
        boxes[is_it - 1].append(boxes[is_it][0])
        boxes.remove(boxes[is_it])
    return boxes


def reject(x, y, w, h, test):
    if w * h > 400:
        return True

    if x < 10 or (x + w) > 54:
        return True

    if h > 30:
        return True

    if not test:
        if y > 20 or y == 0:
            return True
        else:
            return False

    return False


def change(imgfp, modelG, args):
    # Read image
    if args.deepmodel == "True":
        img = Image.open(imgfp)
    else:
        img = imread(imgfp)
    # Get image name
    ori_pth = imgfp.replace(args.root, "")
    n_path = args.out_path + "/" + args.type + args.save_name + "/market1501" + ori_pth
    if not os.path.isfile(n_path):
        n_path = n_path.replace(".jpg.jpg", ".jpg")

    # Deep mode
    if args.deepmodel == "True":
        re1 = enc(img).cuda()
        re1 = re1.unsqueeze(dim=0)
        Blur = transforms.GaussianBlur(25, sigma=(4 ,4))
        re1 = modelG(re1)
        a = vutils.make_grid(re1, normalize=True).cpu()
        a = tt(a)
        a = a.mul(255).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(a)
        im.save(n_path)

    # Normal mode
    else:
        im = deepcopy(img)
        im = Filter(img, args)
        imwrite(im, n_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filtering")
    parser.add_argument(
        "--root", type=str, default="/storage/sjhwang/TransReID-SSL/data/market1501"
    )
    parser.add_argument("--out_path", type=str, default="/storage/sjhwang/gan_result")
    parser.add_argument("--type", type=str, default="cool")
    parser.add_argument("--deepmodel", type=str, default="True")
    parser.add_argument("--para", type=int, default=1)
    parser.add_argument("--save_name", type=str, default="/15")
    parser.add_argument("--number", type=int, default=15)
    parser.add_argument("--m_path", type=str, default="/storage/sjhwang/vedadet/good")
    args = parser.parse_args()

    set(args.save_name, args)
    imgs = "/*.jpg"
    test = args.root + "/bounding_box_test"
    te = glob.glob(test + imgs)

    if args.deepmodel == "True":
        device = torch.device("cuda")
        modelG = ugnet_pixd.GeneratorUNet().cuda()
        modelG.load_state_dict(
            torch.load(args.m_path + "/{}.pth".format(args.number))
        )
        modelG.to(device)
        modelG.eval()
    else:
        device = torch.device("cpu")
        modelG = None
    
    for k in tqdm(te):
        change(k, modelG, args)
