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
import vitpose as pose
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


def get_all(imgfp, root2, args, model_p, transform_pose1, transform_pose2, mse_loss):
    
    # get privacy image
    pri = Image.open(imgfp)
    print(imgfp)
    pri = transform_pose1(pri)
    pri = transform_pose2(pri)
    pri_img = torch.unsqueeze(pri, 0).cuda()
    
    # get origin image
    ori_pth = imgfp.replace(args.root, "")
    n_path = root2 + ori_pth
    if not os.path.isfile(n_path):
        n_path = n_path.replace(".jpg.jpg", ".jpg")
    ori = Image.open(n_path)
    print(n_path)
    ori_img = transform_pose1(ori)
    ori_img = transform_pose2(ori_img)
    ori_img = torch.unsqueeze(ori_img, 0).cuda()
    # get pose error

    pose_ori = model_p(ori_img)
    pose_pri = model_p(pri_img)
    pose_error = mse_loss(pose_ori, pose_pri)

    return pose_error


def main():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--type", type=str, default="noise")
    parser.add_argument("--root", type=str, default="/storage/sjhwang/TransReID-SSL/data/market1501")
    parser.add_argument("--root2", type=str, default="/storage/sjhwang/final")
    parser.add_argument("--check_txt", type=str, default="/storage/sjhwang/TransReID-SSL/human_img_on_gallery.txt")
    args = parser.parse_args()
    model_p, transform_pose1, transform_pose2, mse_loss = pose.ViT_modules()
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

    for value, i in enumerate(ref2_list):
        # measure metrics
        total_pose = 0
        for k in tqdm(lines):
            k = k.strip()
            pose_error = get_all(k, i, args, model_p, transform_pose1, transform_pose2, mse_loss)
            total_pose += torch.mean(pose_error).item()
        total_pose /= len(lines)
        f2 = open(args.root2 + "/" + args.type + "_pose" + ".txt", "a")
        result = i + " " + "pose: " + str(total_pose) + "\n"
        f2.write(result)
        f2.close()
        print(result)


if __name__ == "__main__":
    main()
