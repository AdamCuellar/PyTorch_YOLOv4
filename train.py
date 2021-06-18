import argparse
import os
import math
from copy import deepcopy
import numpy as np
from tqdm import tqdm

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from utils.general import init_seeds, parse_cfg, parse_names, plot_labels, plot_images, check_best, compute_loss
from utils.torch_utils import select_device
from utils.model import Darknet
from utils.datasets import create_dataloader
from test import test

def train():
    init_seeds(args.rng + args.global_rank)
    last = args.logdir + 'last.pt'
    best = args.logdir + 'best.pt'
    cuda = args.device.type != "cpu"
    results_file = os.path.join(args.logdir, "results_{}.txt".format(args.name))

    modelDefs = parse_cfg(args.cfg)
    model = Darknet(modelDefs)
    netParams = model.netParams
    names = parse_names(args.names)

    pretrained = False
    if args.weights:
       pretrained = True
       if args.weights.endswith(".pt"):
          ckpt = torch.load(args.weights, map_location=args.device)  # load checkpoint
          state_dict = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
          model.load_state_dict(state_dict, strict=False)
          print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), args.weights))  # report
       else:
         model.load_darknet_weights(args.weights)

    nominalBS = 64
    accumulate = max(round(nominalBS / args.batch_size), 1)
    netParams["decay"] *= args.batch_size * accumulate / nominalBS

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2.append(v)  # biases
        elif 'Conv2d.weight' in k:
            pg1.append(v)  # apply weight_decay
        else:
            pg0.append(v)  # all else

    if args.adam:
        optimizer = optim.Adam(pg0, lr=netParams["learning_rate"], betas=(netParams['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=netParams["learning_rate"], momentum=netParams['momentum'], nesterov=True)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: (((1 + math.cos(x * math.pi / args.epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    start_epoch, best_map = 0, 0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_map = ckpt['best_map']

        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if args.epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (args.weights, ckpt['epoch'], args.epochs))
            args.epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    model = model.to(args.device)

    # pass fake input to get strides
    with torch.no_grad():
        yolo_strides = model(torch.zeros(1,3,224,224).to(args.device), return_strides=True)

    # DP mode
    if cuda and args.global_rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if args.sync_bn and cuda and args.global_rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.device)
        print('Using SyncBatchNorm()')

    # DDP mode
    if cuda and args.global_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=(args.local_rank))

    # Trainloader
    dataloader, dataset = create_dataloader(args.data, netParams, args.batch_size, local_rank=args.local_rank, world_size=args.world_size)

    # Testloader
    if args.global_rank in [-1, 0]:
        # local_rank is set to -1. Because only the first process is expected to do evaluation.
        testloader, testset = create_dataloader(args.data, netParams, args.batch_size, valid=True, local_rank=args.local_rank, world_size=args.world_size)

    # add model parameters
    model.numClasses = len(names)
    model.names = names
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)

    # Class frequency
    if args.global_rank in [-1, 0]:
        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        plot_labels(labels, save_dir=args.logdir)
        if args.tb_writer:
            args.tb_writer.add_histogram('classes', c, 0)

    # TODO: Check anchors
    #if args.autoanchor:
    #    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    numBatches = len(dataloader)
    maps = np.zeros(len(names))  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    if args.global_rank in [0, -1]:
        print('Image sizes %g train, %g test' % (netParams["height"], netParams["height"]))
        print('Using %g dataloader workers' % dataloader.num_workers)
        print('Starting training for %g epochs...' % args.epochs)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, args.epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(4, device=args.device)  # mean losses
        if args.global_rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if args.global_rank in [-1, 0]:
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(pbar, total=numBatches) # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + numBatches * epoch  # number integrated batches (since train start)
            imgs = imgs.to(args.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= netParams["burn_in"]:
                xi = [0, netParams["burn_in"]]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nominalBS / args.batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, netParams['momentum']])

            # Autocast
            with amp.autocast(enabled=cuda):
                # Forward
                pred = model(imgs)

                # Loss
                loss, loss_items = compute_loss(pred, targets.to(args.device), model)  # scaled by batch_size
                if args.global_rank != -1:
                    loss *= args.world_size  # gradient averaged between devices in DDP mode
                # if not torch.isfinite(loss):
                #     print('WARNING: non-finite loss, ending training ', loss_items)
                #     return results

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()

            # Print
            if args.global_rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, args.epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if ni < 3:
                    f = os.path.join(args.logdir, 'train_batch%g.jpg' % ni)  # filename
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    if args.tb_writer and result is not None:
                        args.tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # DDP process 0 or single-GPU
        if args.global_rank in [-1, 0]:
            # mAP
            final_epoch = epoch + 1 == args.epochs
            if not final_epoch:  # Calculate mAP
                results, maps, times = test(args.data,
                                            imgsz=netParams["height"],
                                            batch_size=args.batch_size,
                                            save_json=final_epoch and "coco" in args.data,
                                            model=model,
                                            dataloader=testloader,
                                            save_dir=args.logdir)

                # Update best mAP
            curr_map = check_best(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
            if curr_map > best_map:
                best_map = curr_map

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

            # Tensorboard
            if args.tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                    args.tb_writer.add_scalar(tag, x, epoch)

                    # Save model
                    with open(results_file, 'r') as f:  # create checkpoint
                        ckpt = {'epoch': epoch,
                                'best_map':curr_map,
                                'training_results': f.read(),
                                'model': model.state_dict(),
                                'optimizer': None if final_epoch else optimizer.state_dict()}

                    # Save last, best and delete
                    torch.save(ckpt, last)
                    if epoch >= (args.epochs - 5):
                        torch.save(ckpt, last.replace('.pt', '_{:03d}.pt'.format(epoch)))
                    if (best_map == curr_map) and not final_epoch:
                        torch.save(ckpt, best)
                    del ckpt
                # end epoch ----------------------------------------------------------------------------------------------------
            # end training

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, default='cfgs/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017_val.data', help='*.data path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--rng', type=int, default=777, help="RNG Seed")
    parser.add_argument('--half', action="store_true", default=False, help="Use half precision")
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--autoanchor', action='store_true', help='Enable autoanchor check')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()

    assert os.path.exists(args.cfg), "CFG {} does not exist".format(args.cfg)
    assert os.path.exists(args.data), "Data file {} does not exist".format(args.data)

    args.device = select_device(args.device, batch_size=args.batch_size)
    args.world_size = 1
    args.global_rank = -1

    # DDP mode
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        args.world_size = dist.get_world_size()
        args.global_rank = dist.get_rank()
        assert args.batch_size % args.world_size == 0, '--batch-size must be multiple of CUDA device count'
        args.batch_size = args.total_batch_size // args.world_size

    args.tb_writer = None
    if args.global_rank in [-1, 0]:
        print('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % args.logdir)
        args.tb_writer = SummaryWriter(log_dir=os.path.join(args.logdir, 'exp', args.name))  # runs/exp

    train()