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



def do_train(args,
             cfg,
             model,
             modelG,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_G,
             optimizer_center,
             scheduler,
             scheduler_G,
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
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # for sum and restore the values.
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # for calculating the results.
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scalerD = amp.GradScaler(enabled=True)
    scalerG = amp.GradScaler(enabled=True)
    L1Loss= nn.L1Loss()

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        modelG.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            #######################
            # update Discriminator
            #######################
            # init
            model.zero_grad()
            
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
            patch = (32, 1, 16, 8)
            valid = torch.ones((img.size(0), 1, 16, 8)).cuda()
            fake = torch.zeros((img.size(0), 1, 16, 8)).cuda()

            # real
            score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
            loss = loss_fn(score, feat, target, target_cam)

            # fake
            with torch.no_grad():
                ch_img = modelG(img)
            score_fake, feat_fake = model(ch_img, target, cam_label=target_cam, view_label=target_view)
            loss_fake = loss_fn(score_fake, feat_fake, target, target_cam)
            rr = dis(img)
            ff = dis(ch_img)
            loss_real_pix = ganloss(rr, valid)
            loss_fake_pix = ganloss(ff, fake)
            loss_gan_d = 0.5*(loss_real_pix + loss_fake_pix)
            loss_D = loss + loss_fake + loss_gan_d
         
            # for wandb log
            D_loss = loss_D.item()
            
            # optimize
            if n_iter % 3 == 0:
                loss_D.backward()
                optimizer.step()

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
            ch_img = modelG(img)
            score_G, feat_G = model(ch_img, target, cam_label=target_cam, view_label=target_view)
            loss_G = loss_fn(score_G, feat_G, target, target_cam)
            l1 = L1Loss(img, ch_img)
            if args.use_lossG:
                to_loss = -loss_G
            elif args.use_L1:
                to_loss = cfg.SOLVER.LAMBDA_L1*l1
            else:
                to_loss = -loss_G + cfg.SOLVER.LAMBDA_L1*l1
                
                ff = dis(ch_img)
                loss_GAN_g = ganloss(ff, valid)
                
                model_pose.eval()
                ori_heat = model_pose(tr_for_pose(img))
                out_heat = model_pose(tr_for_pose(ch_img))
                loss_pose = l2_heat(ori_heat, out_heat)
                to_loss += 100 * loss_pose + loss_GAN_g



            # get loss
            G_loss = -loss_G.item()
            l1_loss = cfg.SOLVER.LAMBDA_L1*l1.item()
            G_total = G_loss + l1_loss
            to_loss.backward()
            optimizer_G.step()
            
            # for center loss
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    if args.made:
                        base_lr = scheduler.get_last_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] reid: {:.3f}, pose: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, loss_pose.item(), base_lr))
                    else:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        
        # wandb logging
        wandb.log({"ReID_loss": D_loss}, commit=False)
        wandb.log({"G_loss": G_loss}, commit=False)
        wandb.log({"L1": l1_loss}, commit=False)
        wandb.log({"G_total": G_total}, commit=False)
        wandb.log({"pose_loss": loss_pose}, commit=False)
        real_image = wandb.Image(img[:1])
        fake_image = wandb.Image(ch_img[:1])
        wandb.log({"real": real_image, 'fake': fake_image})
        # scheduler setting
        if args.made:
            scheduler.step()
            scheduler_G.step()
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

        if epoch % 1 == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'my_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'my_{}.pth'.format(epoch)))
        
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


