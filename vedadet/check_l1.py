from numpy import linalg as LA
import numpy as np
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
import torchvision.transforms as transforms
import lpips
from SSIM_PIL import compare_ssim
import cv2
import argparse
import torch

out_path = "/storage/sjhwang/target_all"
root = "/storage/sjhwang/TransReID-SSL/data/market1501"
imgs = "/*.jpg"
query = root + "/query"
train = root + "/bounding_box_train"
test = root + "/bounding_box_test"
q = glob.glob(query + imgs)
tr = glob.glob(train + imgs)
te = glob.glob(test + imgs)
deep = ["/Super/market1501"]


def get_all(imgfp, root2, args):
    img = Image.open(imgfp)
    img1 = np.array(img)
    ori_pth = imgfp.replace(args.root, "")
    T = transforms.ToTensor()
    a = T(img)
    # privacy image
    n_path = root2 + ori_pth
    if not os.path.isfile(n_path):
        n_path = n_path.replace(".jpg.jpg", ".jpg")
    pri = Image.open(n_path)
    b = T(pri)
    t = transforms.Resize([128, 64])
    
    plt.axis("off")
    plt.title("Original Images")
    plt.imshow(
        np.transpose(vutils.make_grid(a, padding=2, normalize=True).cpu(), (1, 2, 0))
    )
    pri_pil = t(pri)
    pri1 = np.array(pri_pil)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(
        np.transpose(vutils.make_grid(b, padding=2, normalize=True).cpu(), (1, 2, 0))
    )
    get = (img1.flatten() - pri1.flatten()) / (128 * 64 * 3)
    ssim = compare_ssim(img, pri_pil, GPU=False)

    ex_ref = lpips.im2tensor(lpips.load_image(imgfp)).cuda()
    ex_p0 = lpips.im2tensor(lpips.load_image(n_path)).cuda()
    loss_fn_alex = lpips.LPIPS(net="squeeze", spatial=True)
    loss_fn_alex.cuda()
    lpips_alex = loss_fn_alex.forward(ex_ref, ex_p0)

    return (LA.norm(get, 1), ssim, lpips_alex)


def main():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--type", type=str, default="cool")
    parser.add_argument("--root", type=str, default="/storage/sjhwang/TransReID-SSL/data/market1501")
    parser.add_argument("--root2", type=str, default="/storage/sjhwang/real_final")
    parser.add_argument("--check_txt", type=str, default="/storage/sjhwang/TransReID-SSL/human_img_on_gallery.txt")
    args = parser.parse_args()

    imgs = "/*.jpg"
    ref = args.root + "/bounding_box_test"
    re = glob.glob(ref + imgs)
    
    ref2 = args.root2 + "/" + args.type + "/*"
    re2 = glob.glob(ref2)
    ref2_list = []
    for i in re2:
        ref2_list.append(i+"/market1501")
    
    f1 = open(args.check_txt, "r")
    lines = f1.readlines()

    for i in ref2_list:
        # measure metrics
        loss_l1 = 0
        loss_ssim = 0
        loss_alex = 0
        for k in tqdm(lines):
            k = lines[0]
            k = k.strip()
            l1, ssim, alex = get_all(k, i, args)
            loss_l1 += l1
            loss_ssim += ssim
            loss_alex += torch.mean(alex).item()
        loss_l1 /= len(lines)
        loss_ssim /= len(lines)
        loss_alex /= len(lines)
        f2 = open(args.root2 + "/" + args.type + ".txt", "a")
        result = i + " " + "L1: " + str(loss_l1) + " SSIM: " + str(loss_ssim) + " LPIPS: " + str(loss_alex) + "\n"
        f2.write(result)
        f2.close()
        print(result)


if __name__ == "__main__":
    main()
