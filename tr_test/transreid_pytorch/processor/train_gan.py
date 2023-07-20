import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import modifier
import wandb
from torch.autograd import Variable
import torchvision.transforms as transforms

def do_train(
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
             loss_fn,
             perceptual_loss,
             num_query, 
             local_rank,
             model_pose,
             tr_for_pose,
             l2_heat,
             dis):

    # log
    wandblog = 0
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = "cuda"

    # max epoch is normally 120.
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None

    # multi.gpu setting
    if device:
        model.to(local_rank)
        model_color.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
            model_color = torch.nn.parallel.DistributedDataParallel(model_color, device_ids=[local_rank], find_unused_parameters=True)

    # for sum and restore the values.
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    class LabelSmoothingLoss(nn.Module):
        def __init__(self, classes, smoothing=0.0, dim=-1):
            super(LabelSmoothingLoss, self).__init__()
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.cls = classes
            self.dim = dim

        def forward(self, pred, target):
            pred = pred.log_softmax(dim=self.dim)
            with torch.no_grad():
                # true_dist = pred.data.clone()
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(self.smoothing / (self.cls - 1))
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            return -torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    class CrossEntropyLoss(nn.Module):
        def __init__(self, classes, dim=-1):
            super(CrossEntropyLoss, self).__init__()
            self.cls = classes
            self.dim = dim

        def forward(self, pred, target):
            pred = pred.softmax(dim=-1)[torch.arange(pred.size(0)), target]
            return - torch.mean(torch.log(1 - pred))
    CE_loss = CrossEntropyLoss(classes=751)
    #LabelSmoothingLoss(classes=751, smoothing=0.0)
    class MRloss(nn.Module):
        def __init__(self):
            super(MRloss, self).__init__()
        def forward(self, x, target):
            x = x[torch.arange(x.size(0)), target]
            return x.mean()
    MR_loss = MRloss()
    # for calculating the results.
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scalerD = amp.GradScaler(enabled=True)
    scalerG = amp.GradScaler(enabled=True)
    L1Loss= nn.MSELoss()
    def L5loss(x,y):
        return torch.mean(
                ((torch.abs(x-y)**0.5).sum(dim=3).sum(dim=2).sum(dim=1))**2,
                )
    #L1Loss = L5loss
    Gray = nn.Sequential(
        nn.Identity(), ## One Norm 
        #transforms.Grayscale(num_output_channels=1),
        #transforms.GaussianBlur(5, sigma=(2 ,2)),
        transforms.GaussianBlur(35, sigma=(9 ,9)),
    )
    Blur = nn.Sequential( ## G Model
        nn.Identity(),
        #transforms.Grayscale(num_output_channels=1),
    )
    Jitter = nn.Sequential( ## D model
        nn.Identity(),
        #transforms.RandomResizedCrop((256,128), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    )

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.eval()
        model_color.eval()
        modelG.train()
        dis.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            #######################
            # update Discriminator
            #######################
            # init
            model.zero_grad()
            dis.zero_grad()
            

            #optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            """
            with amp.autocast(enabled=True):
                # real
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)
                
                # fake
                ch_img = modelG(img)
                score_fake, feat_fake = model(ch_img, target, cam_label=target_cam, view_label=target_view)
                loss_fake = loss_fn(score_fake, feat_fake, target, target_cam)
                loss_D = loss + loss_fake
            """
            # GAN
            ganloss = torch.nn.MSELoss()

            # real
            # score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
            #loss = loss_fn(score, feat, target, target_cam)

            # fake
            #with torch.no_grad():
            #ch_img = modelG(Blur(img).repeat(1,3,1,1))
            ch_img = modelG(Blur(img))
            #score_fake, feat_fake = model(ch_img.detach(), target, cam_label=target_cam, view_label=target_view)
            #loss_fake = loss_fn(score_fake, feat_fake, target, target_cam)
            rr = dis(Jitter(img))
            ff = dis(ch_img.detach())
            patch = rr.shape
            valid = torch.ones((*patch)).cuda()
            fake = torch.zeros((*patch)).cuda()
            loss_real_pix = ganloss(rr, valid)
            loss_fake_pix = ganloss(ff, fake)
            loss_gan_d = (loss_real_pix + loss_fake_pix)
            
            loss_D = loss_gan_d
         
            # for wandb log
            GAN_D_loss = loss_gan_d.item()
            D_loss = loss_D.item()
            
            # optimize
            loss_D.backward()
            #optimizer.step()
            optimizer_dis.step()

            ########################
            # Update the G
            ########################
            # init
            modelG.zero_grad()
            
            """
            # train G
            with amp.autocast(enabled=True):
                score_G, feat_G = model(ch_img, target, cam_label=target_cam, view_label=target_view)
                loss_G = loss_fn(score_G, feat_G, target, target_cam)
                l1 = L1Loss(img, ch_img)
                to_loss = -loss_G + cfg.SOLVER.LAMBDA_L1*l1
            """
            # train G
            #ch_img = modelG(img)
            score_G, feat_G = model(ch_img.mean(dim=1,keepdims=True).repeat(1,3,1,1), target, cam_label=target_cam, view_label=target_view, adv_train=True)
            loss_G = CE_loss(score_G, target)

            score_G_inv, _ = model( (1-ch_img).mean(dim=1,keepdims=True).repeat(1,3,1,1), target, cam_label=target_cam, view_label=target_view, adv_train=True)
            loss_G_inv = CE_loss(score_G_inv, target)
            #loss_G = loss_fn(score_G, feat_G, target, target_cam)
            score_G_color, feat_G_color = model_color(ch_img, target, cam_label=target_cam, view_label=target_view, adv_train=True)
            loss_G_color = CE_loss(score_G_color, target)
            #loss_G_color = loss_fn(score_G_color, feat_G_color, target, target_cam)

            l1 = L1Loss(
                    Gray(img),
                    Gray(ch_img)
                )
            to_loss = cfg.SOLVER.LAMBDA_L1*l1 / args.w_adv
            if epoch >= args.reid_start:
                to_loss += args.reID_weight * 1*(loss_G)/3 / args.w_adv
            if epoch >= args.reid_start:
                to_loss += args.reID_weight * 1*(loss_G_color)/3 /args.w_adv
            if epoch >= args.reid_start:
                to_loss += args.reID_weight * 1*(loss_G_inv)/3 /args.w_adv
            
            ff = dis(ch_img)
            loss_GAN_g = ganloss(ff, valid)
            
            with torch.no_grad():
                ori_heat = model_pose(tr_for_pose(img))
                out_heat = model_pose(tr_for_pose(ch_img))
            loss_pose = l2_heat(ori_heat, out_heat)
            

            if epoch >= args.pose_start:
                pose_w = args.pose_weight
            else:
                pose_w = 0

            if args.no_pose:
                pose_w = 0
            to_loss += pose_w * loss_pose / args.w_adv + loss_GAN_g



            # get loss
            ReID_G = loss_G.item()
            L1_loss = l1.item()
            if args.no_pose:
                Pose = 0
            else:
                Pose = loss_pose.item()
            to_loss.backward()
            optimizer_G.step()
            
            # for center loss
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            #if isinstance(score, list):
            #    acc = (score[0].max(1)[1] == target).float().mean()
            #else:
            #    acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(0, img.shape[0])
            acc_meter.update(0, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % (log_period * 5) == 0:
                    if args.made:
                        base_lr = scheduler.get_last_lr()[0]
                        if args.use_GAN or args.use_GAN_L1:
                            logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, G_loss: {:.3f}, D_loss: {:.3f}, l1_loss: {:.3f}, Base Lr: {:.2e}"
                                        .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, G_loss, D_loss, l1_loss, base_lr))
                        else:
                            how_many = len(train_loader) / (n_iter+1)
                            logger.info("Epoch[{}] Iter[{}/{}], iter_per_epoch: {:.3f} Discriminator, ReID: {:.3f}, GAN_D_loss: {:.3f}, Generator: ReID_G{:.3f} L1: {:.3f}, Pose: {:.5f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), how_many, loss_meter.avg, GAN_D_loss, ReID_G, L1_loss, Pose, base_lr))
                            wandb.log({"ReID": loss_meter.avg}, commit=False)
                            wandb.log({"GAN_Dis": GAN_D_loss}, commit=False)
                            wandb.log({"L1": L1_loss}, commit=False)
                            wandb.log({"ReID_G": ReID_G}, commit=False)
                            wandb.log({"Pose": Pose}, commit=False)
                            #real_image = wandb.Image(img[:1])
                            #fake_image = wandb.Image(ch_img[:1])
                            real_fake_image = wandb.Image(torch.cat((img[:2], ch_img[:2]), dim=2))
                            torch.save(modelG.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'my_{}_iter_{}.pth'.format(epoch, n_iter)))
                        


                            wandb.log({"real_fake": real_fake_image})
 
                    else:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Discriminator, ReID: {:.3f}, GAN_D_loss: {:.3f}, Generator: ReID_G{:.3f} L1: {:.3f}, Pose: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, GAN_D_loss, ReID_G, L1_loss, Pose, base_lr))
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        
 
        # scheduler setting
        if args.made:
            scheduler.step()
            scheduler_G.step()
            scheduler_dis.step()
        else:
            if cfg.SOLVER.WARMUP_METHOD == 'cosine':
                scheduler.step(epoch)
            else:
                scheduler.step()
        

        # multi-gpu setting        
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        # save generator !!
        if epoch % 10 == 0:
            torch.save(modelG.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'G_0.1' + '_{}.pth'.format(epoch)))

        if epoch % 10 == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


