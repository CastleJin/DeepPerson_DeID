from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from model.vitpose import ViT_modules
from solver import WarmupMultiStepLR
from solver.G_make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from loss.perceptual_loss import VGGPerceptualLoss
from processor.train_gan import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import torch.distributed as dist
import modifier
import wandb
from model import ugnet_pixd


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    # get the arg
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--pose_start", default=1, type=int)
    parser.add_argument("--made", action="store_true")
    parser.add_argument("--use_lossG", action="store_true")
    parser.add_argument("--pose_weight", default = 10, type=float)
    parser.add_argument("--reID_weight", default = 0.3, type=float)
    parser.add_argument("--lr_dis", default=0.0004, type=float)
    parser.add_argument("--use_L1", action="store_true")
    parser.add_argument("--no_pose", action="store_true")
    parser.add_argument("--strong", action="store_true")
    parser.add_argument("--use_GAN", action="store_true")
    parser.add_argument("--use_GAN_L1", action="store_true")
    parser.add_argument("--lr", default=0.0004, type=float)
    parser.add_argument("--lrG", default=0.0004, type=float)
    parser.add_argument('--w-adv', type=float, default=1.0)
    parser.add_argument('--reid-start', type=int, default=1)
    parser.add_argument("--project", default="change_lr", type=str)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--dis", default=0.0004, type=float)
    args = parser.parse_args()

    # set config file
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # wandb init configuration
    if args.strong:
        wandb_project = "strong-" + args.project
    else:
        wandb_project = args.project
    if args.name is "":
        wandb_name = "lr-{}-lrG-{}-L1-{}-epoch-{}-batch-{}-server-9".format(
            args.lr,
            args.lrG,
            cfg.SOLVER.LAMBDA_L1,
            cfg.SOLVER.MAX_EPOCHS,
            cfg.SOLVER.IMS_PER_BATCH,
        )
    else:
        wandb_name = args.name

    if args.made:
        wandb_config = {
            "G_optimizer": "Adam, lr={}, betas=(0.5, 0.999)".format(args.lrG),
            "ReID_optimizer": "Adam, lr={}, betas=(0.5, 0.999)".format(args.lr),
            "lambda_l1": "{}".format(cfg.SOLVER.LAMBDA_L1),
            "epoch": "{}".format(cfg.SOLVER.MAX_EPOCHS),
            "batch": "{}".format(cfg.SOLVER.IMS_PER_BATCH),
        }
    else:
        wandb_config = {
            "G_optimizer": "Adam, lr=0.0002, betas=(0.5, 0.999)",
            "ReID_optimizer": "Sgd, lr=0.0004, cosine warmup",
            "lambda_l1": "{}".format(cfg.SOLVER.LAMBDA_L1),
            "epoch": "{}".format(cfg.SOLVER.MAX_EPOCHS),
            "batch": "{}".format(cfg.SOLVER.IMS_PER_BATCH),
        }
    wandb.init(config=wandb_config, project=wandb_project, name=wandb_name)

    # set seed
    # for random seed.
    # manualSeed = random.randint(1, 10000)
    set_seed(cfg.SOLVER.SEED)
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    try:
        os.makedirs(output_dir)
    except:
        pass

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            #  logger.info(config_str)

    # set multi-gpu setting
    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    logger.info("Running with config:\n{}".format(cfg))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    perceptual_loss = VGGPerceptualLoss().cuda()
    # dataset setting.
    (
        train_loader,
        train_loader_normal,
        val_loader,
        num_query,
        num_classes,
        camera_num,
        view_num,
    ) = make_dataloader(cfg)

    # model setting.
    model = make_model(
        cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    
    cfg.PRETRAIN_PATH: 'model/vit_small_cfs_ics/transformer_120.pth'
    model_color = make_model(
        cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.eval()
    model_color.eval()
    modelG = ugnet_pixd.GeneratorUNet().cuda()
    dis = ugnet_pixd.Discriminator().cuda()
    model_pose, tr_for_pose, l2_heat = ViT_modules()
    model_pose.eval()
    if args.strong:
        path_to_strong_G = "/storage/sjhwang/tr_test/transreid_pytorch/log/transreid/L1/transformerG_0.1_120.pth"
        modelG.load_state_dict(torch.load(path_to_strong_G))
    device = torch.device("cuda")
    modelG.to(device)

    ## init the generator
    # torch.nn.init.normal_(G.weight)
    # torch.nn.init.xavier_uniform_(G.weight)
    # torch.nn.init.kaiming_uniform_(layer.weight)

    # ReID Loss
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # optimizer
    if args.made:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizer_G = torch.optim.Adam(
            modelG.parameters(), lr=args.lrG, betas=(0.5, 0.999)
        )
        optimizer_dis = torch.optim.Adam(
            dis.parameters(), lr=args.lr_dis, betas=(0.5, 0.999)
        )
        optimizer_center = None
    else:
        optimizer, optimizer_G, optimizer_center = make_optimizer(
            cfg, model, modelG, center_criterion
        )

    # scheduler
    if args.made:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.995)
        scheduler_dis = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_dis, gamma=0.995
        )
    else:
        if cfg.SOLVER.WARMUP_METHOD == "cosine":
            logger.info("===========using cosine learning rate=======")
            scheduler = create_scheduler(cfg, optimizer)
        else:
            logger.info("===========using normal learning rate=======")
            scheduler = WarmupMultiStepLR(
                optimizer,
                cfg.SOLVER.STEPS,
                cfg.SOLVER.GAMMA,
                cfg.SOLVER.WARMUP_FACTOR,
                cfg.SOLVER.WARMUP_EPOCHS,
                cfg.SOLVER.WARMUP_METHOD,
            )
        scheduler_G = None

    do_train(
        args,
        cfg,
        model,
        model_color,
        modelG,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_G,
        optimizer_dis,
        optimizer_center,
        scheduler,
        scheduler_G,
        scheduler_dis,
        loss_func,
        perceptual_loss,
        num_query,
        args.local_rank,
        model_pose,
        tr_for_pose,
        l2_heat,
        dis
    )
    wandb.finish()
    #  print(cfg.OUTPUT_DIR)
    #  print(cfg.MODEL.PRETRAIN_PATH)
