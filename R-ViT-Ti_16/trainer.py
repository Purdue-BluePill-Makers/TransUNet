import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, LogCoshDiceLoss
from torchvision import transforms
from lion import Lion
import wandb

def trainer_lpcv(args, model, snapshot_path):
    from datasets.dataset_lpcv import LPCV_dataset
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = LPCV_dataset(data_dir = args.root_path, gt_data_dir = args.list_dir, split="train")
    
    db_val = LPCV_dataset(data_dir=args.root_path, gt_data_dir=args.list_dir, split = "val")
    
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    ce_loss = CrossEntropyLoss()

    dice_loss_functions = {
        'DICE' : DiceLoss(num_classes),
        'LOGCOSHDICE' : LogCoshDiceLoss(num_classes),
    }
    dice_loss = dice_loss_functions[args.loss_function]

    optimizers = {
        'SGD' : optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.001),
        'ADAM' : optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.001),
        'ADAMW' : optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.001),
        'LION' : Lion(model.parameters(), lr=base_lr, weight_decay=0.001),
    }
    optimizer = optimizers[args.optimizer]

    # writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    best_val_loss = float('inf')
    for epoch_num in iterator:
        model.train()
        epoch_total_loss = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_dice = 0.0
        epoch_class_loss_dice = [0.0] * num_classes
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            label_batch = label_batch
            label_batch = label_batch.squeeze(1)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice, class_wise_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_dice + 0.5 * loss_ce

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)
            # writer.add_scalar('info/dice_loss', loss_dice, iter_num)
            # writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            
            epoch_loss_ce += loss_ce.item()
            epoch_loss_dice += loss_dice.item()
            epoch_total_loss += loss.item()
            epoch_class_loss_dice = [epoch_loss + iter_loss for epoch_loss, iter_loss in zip(epoch_class_loss_dice, class_wise_dice)]
            
            wandb.log({"loss_ce": loss_ce,
                       "loss_dice": loss_dice,
                       "total_loss": loss,
                       "class_wise_dice": class_wise_dice,
                       "iteration": iter_num})

            logging.info('iteration %d : loss : %f, loss_dice: %f loss_ce: %f' % (iter_num, loss.item(), loss_dice.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:3, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                # writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                outputs = outputs[1, ...] * 50
                labs = label_batch[1, ...].unsqueeze(0)
                labs = labs * 50
                # writer.add_image('train/GroundTruth', labs * 50, iter_num)
                
                wandb.log({
                    "inputs" : wandb.Image(image),
                    "gt": wandb.Image(labs),
                    "prediction": wandb.Image(outputs.float())
                    })
        
        model.eval()
        
        epoch_val_loss_ce = 0.0
        epoch_val_loss_dice = 0.0
        epoch_val_total_loss = 0.0
        epoch_val_class_loss_dice = [0.0] * num_classes
        with torch.no_grad():
            
            for i_batch, sampled_batch in enumerate(val_loader):
                val_image_batch, val_label_batch = sampled_batch['image'], sampled_batch['label']
                val_image_batch, val_label_batch = val_image_batch.cuda(), val_label_batch.cuda()
                val_outputs = model(val_image_batch)
                val_label_batch = val_label_batch
                val_label_batch = val_label_batch.squeeze(1)
                val_loss_ce = ce_loss(val_outputs, val_label_batch[:].long())
                val_loss_dice, val_class_wise_dice = dice_loss(val_outputs, val_label_batch, softmax=True)
                val_loss = 0.5 * val_loss_ce + 0.5 * val_loss_dice
                
                epoch_val_loss_ce += val_loss_ce.item()
                epoch_val_loss_dice += val_loss_dice.item()
                epoch_val_total_loss += val_loss.item()
                epoch_class_loss_dice = [epoch_loss + iter_loss for epoch_loss, iter_loss in zip(epoch_class_loss_dice, class_wise_dice)]
                
                wandb.log({
                "val_loss_ce": val_loss_ce,
                "val_loss_dice": val_loss_dice,
                "val_total_loss": val_loss,
                "val_class_wise_dice": val_class_wise_dice,
                "val_iteration": iter_num})
                
                val_image = val_image_batch[1, 0:3, :, :]
                val_image = (val_image - val_image.min()) / (val_image.max() - val_image.min())
                val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1, keepdim=True)
                val_outputs = val_outputs[1, ...] * 50
                val_labs = val_label_batch[1, ...].unsqueeze(0) * 50
                
                wandb.log({
                "val_inputs" : wandb.Image(val_image),
                "val_gt": wandb.Image(val_labs),
                "val_prediction": wandb.Image(val_outputs.float())
                })
                
        val_loss /= len(val_loader)
        # 에폭별 평균 손실 계산
        avg_epoch_loss_ce = epoch_loss_ce / len(trainloader)
        avg_epoch_loss_dice = epoch_loss_dice / len(trainloader)
        avg_epoch_total_loss = epoch_total_loss / len(trainloader)
        avg_epoch_class_wise_dice = [x / len(trainloader) for x in epoch_class_loss_dice]
        
        avg_val_epoch_loss_ce = epoch_val_loss_ce / len(val_loader)
        avg_val_epoch_loss_dice = epoch_val_loss_dice / len(val_loader)
        avg_val_epoch_total_loss = epoch_val_total_loss / len(val_loader)
        avg_val_epoch_class_wise_dice = [x / len(val_loader) for x in epoch_val_class_loss_dice]
        # 에폭별 로깅
        wandb.log({
            "avg_loss_ce": avg_epoch_loss_ce,
            "avg_loss_dice": avg_epoch_loss_dice,
            "avg_total_loss": avg_epoch_total_loss,
            "avg_val_loss_ce": avg_val_epoch_loss_ce,
            "avg_val_loss_dice": avg_val_epoch_loss_dice,
            "avg_val_total_loss": avg_val_epoch_total_loss,
            "epoch": epoch_num
        })

        logging.info('Epoch %d : Avg Loss: %f, Avg Loss_CE: %f, Avg Loss_Dice : %f' % (epoch_num, avg_epoch_total_loss, avg_epoch_loss_ce, avg_epoch_loss_dice))
        print("train class dice", avg_epoch_class_wise_dice)
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Save best model to {}".format(save_mode_path))
        
        logging.info('Epoch %d : Val Avg Loss: %f, Val Avg Loss_CE: %f, Val Avg Loss_Dice : %f' % (epoch_num, avg_val_epoch_total_loss, avg_val_epoch_loss_ce, avg_val_epoch_loss_dice))
        print("val class dice", avg_val_epoch_class_wise_dice)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    # writer.close()
    return "Training Finished!"
