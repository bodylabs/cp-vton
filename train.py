#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint, WUTON
from torch.autograd import Variable
import torch.autograd as autograd

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images
from p2p_discriminator import define_D

lambda_gp = 10
device = torch.device("cuda" if 1 else "cpu")

Tensor = torch.cuda.FloatTensor if 1 else torch.FloatTensor

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 300000)
    parser.add_argument("--decay_step", type=int, default = 300000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt




def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1, 6, 4).fill_(1.0), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

############### need to modify the cp_dataset.py before using
def train_wuton(opt, train_loader, model_wuton, netD, board):
    # model_gmm.cuda()
    # model_gmm.train()

    # model_tom.cuda()
    # model_tom.train()

    model_wuton.cuda()
    model_wuton.train()

    BCE_stable = torch.nn.BCEWithLogitsLoss()
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()

    # netD = define_D(3, 64, 'n_layers', 5, norm='batch', init_type='normal', gpu_ids=[0])

    netD.cuda()
    netD.train()

    optimizer_G = torch.optim.Adam(model_wuton.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999), )


    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()
        im_g = inputs['grid_image'].cuda()
        c = inputs['cloth'].cuda()
        c_unpaired = inputs['c_unpaired'].cuda()
        dilated_upper_wuton = inputs['dilated_upper_wuton'].cuda()
        im_c =  inputs['parse_cloth'].cuda()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        for p in netD.parameters():
            p.requires_grad_(True)  # freeze D

        for p in model_wuton.parameters():
            p.requires_grad_(False)  # reset G

        
        ########unpaired
        outputs_unpaired, grid_unpaired, theta_unpaired = model_wuton(c_unpaired, dilated_upper_wuton)
        outputs_unpaired = F.tanh(outputs_unpaired)


        y = torch.ones(outputs_unpaired.size()[0], 1, 6, 4).to(device) ########all 1


        # Discriminator loss
        optimizer_D.zero_grad()
        y_pred = netD(im)
        y_pred_fake_D = netD(outputs_unpaired.detach()) # discriminator        

        gradient_penalty = compute_gradient_penalty(netD, im.data.to(device), outputs_unpaired.data.to(device))
        relativistic_loss_d = BCE_stable(y_pred - y_pred_fake_D, y)
        loss_d = relativistic_loss_d + lambda_gp * gradient_penalty
        loss_d.backward()
        optimizer_D.step()


        if (step+1) % opt.display_count == 0:
            # board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric_d', loss_d.item(), step+1)
            board.add_scalar('relativistic_loss_d', relativistic_loss_d.item(), step+1)
            t = time.time() - iter_start_time
            print('discriminator step: %8d, time: %.3f, loss_d: %.4f, relativistic_loss_d: %.4f' 
                    % (step+1, t, loss_d.item(), relativistic_loss_d.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(netD, os.path.join(opt.checkpoint_dir, opt.name, 'netD_step_%06d.pth' % (step+1)))



        if (step+1) % 1 == 0:

            # ---------------------
            #  Train generator
            # # ---------------------
            for p in netD.parameters():
                p.requires_grad_(False)  # freeze D

            for p in model_wuton.parameters():
                p.requires_grad_(True)  # reset G


            #########paired
            outputs, grid, theta = model_wuton(c, dilated_upper_wuton)
            warped_cloth = F.grid_sample(c, grid, padding_mode='border')
            warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
            outputs = F.tanh(outputs)

            # Generator loss (You may want to resample again from real and fake data)
            optimizer_G.zero_grad()
            loss_warp_l1 = criterionL1(warped_cloth, im_c)    
            loss_l1 = criterionL1(outputs, im)
            loss_vgg = criterionVGG(outputs, im)


            outputs_unpaired_g, grid_unpaired, theta_unpaired = model_wuton(c_unpaired, dilated_upper_wuton)
            warped_cloth_unpaired = F.grid_sample(c_unpaired, grid_unpaired, padding_mode='border')
            outputs_unpaired_g = F.tanh(outputs_unpaired_g)
            y_pred_G = netD(im)
            y_pred_fake_G = netD(outputs_unpaired_g) # generator
            relativistic_loss_g = BCE_stable(y_pred_fake_G - y_pred_G, y)
            loss_g = relativistic_loss_g + loss_warp_l1 + loss_l1 + loss_vgg

            visuals = [[c, warped_cloth, im_c], 
                       [dilated_upper_wuton, outputs, im],
                       [c_unpaired, warped_cloth_unpaired, outputs_unpaired_g]]




            loss_g.backward()

            optimizer_G.step()
                
            if (step+1) % opt.display_count == 0:
                board_add_images(board, 'combine', visuals, step+1)
                board.add_scalar('metric_g', loss_g.item(), step+1)
                board.add_scalar('relativistic_loss_g', relativistic_loss_g.item(), step+1)
                board.add_scalar('warp_L1', loss_warp_l1.item(), step+1)
                board.add_scalar('final_L1', loss_l1.item(), step+1)
                board.add_scalar('VGG', loss_vgg.item(), step+1)
                t = time.time() - iter_start_time
                print('generator step: %8d, time: %.3f, loss_g: %.4f, warp_l1: %.4f, final_l1: %.4f, vgg: %.4f, relativistic_loss_g: %.4f' 
                        % (step+1, t, loss_g.item(), 
                        loss_warp_l1.item(), loss_l1.item(), loss_vgg.item(), relativistic_loss_g.item()), flush=True)

            if (step+1) % opt.save_count == 0:
                save_checkpoint(model_wuton, os.path.join(opt.checkpoint_dir, opt.name, 'wuton_step_%06d.pth' % (step+1)))

def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(logdir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        model_wuton = WUTON(opt, 3, 3, 5, ngf=16, norm_layer=nn.InstanceNorm2d)
        netD = define_D(3, 64, 'n_layers', 5, norm='batch', init_type='normal', gpu_ids=[0])
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            print('using loaded model')
            load_checkpoint(model_wuton, opt.checkpoint)
            load_checkpoint(netD, opt.checkpoint.replace('wuton_final', 'netD_final'))
        train_wuton(opt, train_loader, model_wuton, netD, board)
        save_checkpoint(model_wuton, os.path.join(opt.checkpoint_dir, opt.name, 'wuton_final.pth'))
        save_checkpoint(netD, os.path.join(opt.checkpoint_dir, opt.name, 'netD_final.pth'))

        # raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
        
  
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
