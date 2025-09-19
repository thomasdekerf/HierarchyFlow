import numpy as np
import os
from numpy import histogram


import torch
import torch.distributed as dist
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from model.losses import VGGLoss
from model.network.hf import HierarchyFlow
from model.utils.dataset import get_dataset
from model.utils.sampler import DistributedGivenIterationSampler, DistributedTestSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    CosineSimilarity,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import logging
from model.utils.log_helper import init_log

init_log('pytorch hierarchy flow')
global_logger = logging.getLogger('pytorch hierarchy flow')


def save_checkpoint(state, filename):
    torch.save(state, filename+'.pth.tar')

def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['step']

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class Trainer():
    def __init__(self, cfg, local_rank, world_size):
        self.cfg = cfg
        self.rank = local_rank
        self.world_size = world_size
        
        model = HierarchyFlow(self.cfg.network.pad_size, self.cfg.network.in_channel, self.cfg.network.out_channels, self.cfg.network.weight_type)
        model.cuda(self.rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])

        if self.rank == 0:
            global_logger.info(self.cfg)
            global_logger.info(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)

        if self.cfg.eval_mode or (self.cfg.resume and os.path.isfile(self.cfg.load_path)):
            self.model, self.optimizer, self.resumed_step = load_checkpoint(self.cfg.load_path, model, optimizer)
            global_logger.info("=> loaded checkpoint '{}' with current step {}".format(self.cfg.load_path, self.resumed_step))
        else:
            self.model = model
            self.optimizer = optimizer
            self.resumed_step = -1

        if self.cfg.lr_scheduler.type == 'cosine':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.cfg.max_iter, self.cfg.lr_scheduler.eta_min)
        else:
            raise RuntimeError('lr_scheduler {} is not implemented'.format(self.cfg.lr_scheduler))

        self.criterion = VGGLoss(self.cfg.loss.vgg_encoder).cuda(self.rank)
        
        if self.rank == 0:
            self.logger = SummaryWriter(os.path.join(self.cfg.output, self.cfg.task_name, 'runs'))

        early_cfg = getattr(self.cfg, 'early_stopping', {}) or {}
        self.monitor_metric = early_cfg.get('metric', 'ssim').lower()
        default_mode = 'min' if self.monitor_metric in ['lpips', 'fid', 'kid'] else 'max'
        self.early_mode = early_cfg.get('mode', default_mode).lower()
        self.early_min_delta = early_cfg.get('min_delta', 0.0)
        self.early_patience = early_cfg.get('patience', 10)
        self.early_enabled = early_cfg.get('enabled', True)
        self.lpips_net = early_cfg.get('lpips_net', 'alex')

        self.best_metric = None
        self.best_step = None
        self.early_wait = 0

    def train(self):
        train_dataset = get_dataset(self.cfg.dataset.train)
        train_sampler = DistributedGivenIterationSampler(train_dataset,
            self.cfg.max_iter, self.cfg.dataset.train.batch_size, world_size=self.world_size, rank=self.rank, last_iter=-1)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.dataset.train.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            sampler=train_sampler)

        eval_freq = getattr(self.cfg, 'eval_freq', self.cfg.max_iter)
        stop_training = False
        for batch_id, batch in enumerate(train_loader):
            self.train_iter(batch_id, batch)

            if (batch_id + 1) % eval_freq == 0:
                stop_training = self._perform_evaluation(batch_id + 1)
                if stop_training:
                    break

        # Final evaluation if last batch didn't trigger it
        if not stop_training and (batch_id + 1) % eval_freq != 0:
            self._perform_evaluation(batch_id + 1)

    def _perform_evaluation(self, step):
        metrics = self.eval(step)
        stop_training = False
        stop_tensor = torch.zeros(1, device=torch.device('cuda', self.rank))
        if self.rank == 0:
            stop_training = self._handle_eval_metrics(metrics, step)
            stop_tensor.fill_(1 if stop_training else 0)
        dist.broadcast(stop_tensor, src=0)
        stop_training = bool(stop_tensor.item())
        return stop_training

    def _handle_eval_metrics(self, metrics, step):
        if metrics is None:
            return False

        monitored_value = metrics.get(self.monitor_metric)
        if monitored_value is None:
            global_logger.warning(
                "Monitored metric '%s' was not found in evaluation metrics. Available keys: %s",
                self.monitor_metric,
                list(metrics.keys()),
            )
            return False

        improved = self._is_improvement(monitored_value)
        if improved:
            self._save_best_checkpoint(step)
            self.early_wait = 0
        else:
            self.early_wait += 1

        if not self.early_enabled:
            return False

        if self.early_wait >= self.early_patience:
            global_logger.info(
                "Early stopping triggered at step %d with %s=%.6f.",
                step,
                self.monitor_metric.upper(),
                monitored_value,
            )
            return True

        return False

    def _is_improvement(self, value):
        if self.best_metric is None:
            self.best_metric = value
            return True

        if self.early_mode == 'max':
            if value > self.best_metric + self.early_min_delta:
                self.best_metric = value
                return True
        else:
            if value < self.best_metric - self.early_min_delta:
                self.best_metric = value
                return True
        return False

    def _save_best_checkpoint(self, step):
        self.best_step = step
        save_checkpoint(
            {
                'step': step,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            },
            os.path.join(self.cfg.output, self.cfg.task_name, 'model_save', 'best_model'),
        )
        global_logger.info(
            "New best model at step %d with %s=%.6f. Saved checkpoint to %s",
            step,
            self.monitor_metric.upper(),
            self.best_metric,
            os.path.join(self.cfg.output, self.cfg.task_name, 'model_save', 'best_model.pth.tar'),
        )

    def eval(self, step=None):
        test_dataset = get_dataset(self.cfg.dataset.test)
        test_sampler = DistributedTestSampler(test_dataset, world_size=self.world_size, rank=self.rank)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.dataset.test.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            sampler=test_sampler)
        self.model.eval()
        if self.rank == 0:
            psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda(self.rank)
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda(self.rank)
            fid_metric = FrechetInceptionDistance().cuda(self.rank)
            kid_metric = KernelInceptionDistance().cuda(self.rank)
            mra_metric = CosineSimilarity(dim=1).cuda(self.rank)
            lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=self.lpips_net).cuda(self.rank)
        with torch.no_grad():
            for batch_id, batch in enumerate(test_loader):
                content_images = batch[0].cuda(self.rank)
                style_images = batch[1].cuda(self.rank)
                names = batch[2]
                outputs = self.model(content_images, style_images)
                outputs = torch.clamp(outputs, 0, 1)

                if self.rank == 0:
                    psnr_metric.update(outputs, style_images)
                    ssim_metric.update(outputs, style_images)
                    fid_metric.update(style_images, real=True)
                    fid_metric.update(outputs, real=False)
                    kid_metric.update(style_images, real=True)
                    kid_metric.update(outputs, real=False)
                    mra_metric.update(outputs.view(outputs.size(0), -1), style_images.view(style_images.size(0), -1))
                    lpips_metric.update(outputs * 2 - 1, style_images * 2 - 1)

                outputs_cpu = outputs.cpu()
                for idx in range(len(outputs_cpu)):
                    output_name = os.path.join(self.cfg.output, self.cfg.task_name, 'eval_results', 'pred', names[idx])
                    save_image(outputs_cpu[idx].unsqueeze(0), output_name)
                    if idx == 0:
                        output_name = os.path.join(self.cfg.output, self.cfg.task_name, 'eval_results', 'cat_img', names[idx])
                        output_images = torch.stack((content_images[idx].cpu(), style_images[idx].cpu(), outputs_cpu[idx]), 0)
                        save_image(output_images, output_name, nrow=1)
                if self.rank == 0 and batch_id % 10 == 1:
                    global_logger.info('predicting {}th batch...'.format(batch_id))
        if self.rank == 0:
            psnr_val = psnr_metric.compute().item()
            ssim_val = ssim_metric.compute().item()
            fid_val = fid_metric.compute().item()
            kid_mean, _ = kid_metric.compute()
            mra_val = mra_metric.compute().item()
            lpips_val = lpips_metric.compute().item()
            log_step = step if step is not None else 0
            self.logger.add_scalar("PSNR", psnr_val, log_step)
            self.logger.add_scalar("SSIM", ssim_val, log_step)
            self.logger.add_scalar("FID", fid_val, log_step)
            self.logger.add_scalar("KID", kid_mean.item(), log_step)
            self.logger.add_scalar("MRA", mra_val, log_step)
            self.logger.add_scalar("LPIPS", lpips_val, log_step)
            global_logger.info(
                'PSNR: {:.4f}, SSIM: {:.4f}, FID: {:.4f}, KID: {:.4f}, MRA: {:.4f}, LPIPS: {:.4f}'.format(
                    psnr_val, ssim_val, fid_val, kid_mean.item(), mra_val, lpips_val
                )
            )
            global_logger.info('Save predictions to {}\nDone.'.format(os.path.join(self.cfg.output, self.cfg.task_name, 'eval_results')))

        self.model.train()

        if self.rank == 0:
            return {
                'psnr': psnr_val,
                'ssim': ssim_val,
                'fid': fid_val,
                'kid': kid_mean.item(),
                'mra': mra_val,
                'lpips': lpips_val,
            }
        return None

    def train_iter(self, batch_id, batch):
        content_images = batch[0].cuda(self.rank)
        style_images = batch[1].cuda(self.rank)

        outputs = self.model(content_images, style_images)
        outputs = torch.clamp(outputs, 0, 1)

        loss_c, loss_s = self.criterion(content_images, style_images, outputs, self.cfg.loss.k)
        loss_c = loss_c.mean()
        loss_s = loss_s.mean()
        loss = loss_c + self.cfg.loss.weight * loss_s

        torch.distributed.barrier()

        loss = reduce_mean(loss, self.world_size)
        loss_c = reduce_mean(loss_c, self.world_size)
        loss_s = reduce_mean(loss_s, self.world_size)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        if self.rank == 0:
            current_lr = self.lr_scheduler.get_lr()[0]
            self.logger.add_scalar("current_lr", current_lr, batch_id + 1)
            self.logger.add_scalar("loss_c", loss_c.item(), batch_id + 1)
            self.logger.add_scalar("loss_s", loss_s.item(), batch_id + 1)
            self.logger.add_scalar("loss", loss.item(), batch_id + 1)

        if self.rank == 0 and batch_id % self.cfg.print_freq == 0:
            global_logger.info('batch: {}, style_loss: {}, content_loss: {}, loss: {}'.format(batch_id, loss_s.item(), loss_c.item(), loss.item()))
            output_name = os.path.join(self.cfg.output, self.cfg.task_name, 'img_save', str(batch_id)+'.jpg')
            output_images = torch.cat((content_images.cpu(), style_images.cpu(), outputs.cpu()), 0)
            save_image(output_images, output_name, nrow=1)

        if self.rank == 0 and batch_id % self.cfg.save_freq == 0:
            save_checkpoint({
                'step':batch_id,
                'state_dict':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict()
                },os.path.join(self.cfg.output, self.cfg.task_name, 'model_save', str(batch_id)+ '.ckpt'))