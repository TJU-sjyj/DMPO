import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time
import math
import sys
import os
import torch.nn as nn
from utils import *
import torch.distributed as dist

def sample_configs(choices, is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False):

    config = {}
    depth = choices['depth']

    if is_visual_prompt_tuning == False and is_adapter == False and is_LoRA == False and is_prefix==False:
        visual_prompt_depth = random.choice(choices['visual_prompt_depth'])
        lora_depth = random.choice(choices['lora_depth'])
        adapter_depth = random.choice(choices['adapter_depth'])
        prefix_depth = random.choice(choices['prefix_depth'])
        config['visual_prompt_dim'] = [random.choice(choices['visual_prompt_dim']) for _ in range(visual_prompt_depth)] + [0] * (depth - visual_prompt_depth)
        config['lora_dim'] = [random.choice(choices['lora_dim']) for _ in range(lora_depth)] + [0] * (depth - lora_depth)
        config['adapter_dim'] = [random.choice(choices['adapter_dim']) for _ in range(adapter_depth)] + [0] * (depth - adapter_depth)
        config['prefix_dim'] = [random.choice(choices['prefix_dim']) for _ in range(prefix_depth)] + [0] * (depth - prefix_depth)

    else:
        if is_visual_prompt_tuning:
            config['visual_prompt_dim'] = [choices['super_prompt_tuning_dim']] * (depth)
        else:
            config['visual_prompt_dim'] = [0] * (depth)
        
        if is_adapter:
             config['adapter_dim'] = [choices['super_adapter_dim']] * (depth)
        else:
            config['adapter_dim'] = [0] * (depth)

        if is_LoRA:
            config['lora_dim'] = [choices['super_LoRA_dim']] * (depth)
        else:
            config['lora_dim'] = [0] * (depth)

        if is_prefix:
            config['prefix_dim'] = [choices['super_prefix_dim']] * (depth)
        else:
            config['prefix_dim'] = [0] * (depth)
        
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False,lmda=None,output_dir=None, use_alpha=False):
    model.train()
    model = model.cuda()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))
        
    cnt = 0
    w1 = 0.0
    w2 = 0.0
    w3 = 0.0
    w4 = 0.0
    beta1 = 0.0
    beta2 = 0.0
    beta3 = 0.0
    beta4 = 0.0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # sample random config
        if mode == 'super':
            # sample
            config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=is_prefix)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
        else:
            outs = model(samples)
            out1, out2, out3, out4 = outs
            if use_alpha:
                loss1 = criterion(None, out1, targets, epoch)
                loss2 = criterion(out1, out2, targets, epoch)
                loss3 = criterion(out2, out3, targets, epoch)
                loss4 = criterion(out3, out4, targets, epoch)
            else:
                loss1 = criterion(out1, targets)
                loss2 = criterion(out2, targets)
                loss3 = criterion(out3, targets)
                loss4 = criterion(out4, targets)

            loss = lmda.get(0) * loss1 + lmda.get(1) * loss2 + lmda.get(2) * loss3 + lmda.get(3) * loss4
            
            if output_dir is not None and utils.is_main_process():
                cnt += 1
                w1 += loss1.item()
                w2 += loss2.item()
                w3 += loss3.item()
                w4 += loss4.item()
                beta1 += model.module.beta1.detach()
                beta2 += model.module.beta2.detach()
                beta3 += model.module.beta3.detach()
                beta4 += model.module.beta4.detach()
                txt = f"Epoch: {epoch}, Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, Loss3: {loss3.item():.4f}, Loss4: {loss4.item():.4f}\n"
                with (output_dir / "Loss_log.txt").open("a") as f:
                    f.write(txt)
                txt = f"Epoch: {epoch}, Beta1: {model.module.beta1.detach():.4f}, Beta2: {model.module.beta2.detach():.4f}, Beta3: {model.module.beta3.detach():.4f}, Beta4: {model.module.beta4.detach():.4f}\n"
                with (output_dir / "Beta_log.txt").open("a") as f:
                    f.write(txt)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            if lmda.gradnorm.use_gn:
                loss.backward(retain_graph=True)
                lmda.update_gn([lmda.get(0) * loss1, lmda.get(1) * loss2, lmda.get(2) * loss3, lmda.get(3) * loss4])
            else:
                loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    if output_dir is not None and utils.is_main_process():
        w1 = w1 / float(cnt)
        w2 = w2 / float(cnt)
        w3 = w3 / float(cnt)
        w4 = w4 / float(cnt)
        b1 = beta1 / float(cnt)
        b2 = beta2 / float(cnt)
        b3 = beta3 / float(cnt)
        b4 = beta4 / float(cnt)

        txt = f"Avg Loss1: {w1:.4f}, Avg Loss2: {w2:.4f}, Avg Loss3: {w3:.4f}, Avg Loss4: {w4:.4f}\n"
        with (output_dir / "Loss_log.txt").open("a") as f:
            f.write(txt)
        txt = f"Avg Beta1: {b1:.4f}, Avg Beta2: {b2:.4f}, Avg Beta3: {b3:.4f}, Avg Beta4: {b4:.4f}\n"
        with (output_dir / "Beta_log.txt").open("a") as f:
            f.write(txt)
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=False)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def exit_evaluate(model, train_dl, test_dl, dataset_name, epoch, device, dirname, amp=True, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False):
#     tester = Tester(model)
#     model = model.cuda()
#     val_pred, val_target = tester.calc_logit(train_dl, early_break=True)
#     test_pred, test_target = tester.calc_logit(test_dl, early_break=False)
#     each_exit = False
#     if not os.path.exists(f"exit_detail_slr/{dataset_name}/{dirname}"):
#         os.makedirs(f"exit_detail_slr/{dataset_name}/{dirname}")
#     with open("exit_detail_slr/{}/{}/evaluate_repvit_fullfintune_{}_{}.txt".format(dataset_name, dirname, dataset_name, epoch), 'w') as fout:
#         probs_list = generate_distribution(each_exit=each_exit)
#         cnt = 0
#         acc70 = 0.0
#         acc50 = 0.0
#         for probs in probs_list:
#             print('\n*****************')
#             print(probs)
#             acc_val,  T = tester.dynamic_eval_find_threshold(val_pred, val_target, probs)
#             print(T)
#             acc_test, acc_each_stage = tester.dynamic_eval_with_threshold(test_pred, test_target, T)
#             print('valid acc: {:.3f}, test acc: {:.3f}'.format(acc_val, acc_test))
#             print('acc of each exit: {}'.format(acc_each_stage))
#             fout.write('\n*****************\n')
#             fout.write(str(probs.tolist()))
#             fout.write('\n')
#             fout.write(str(T.tolist()))
#             fout.write('\n')
#             fout.write('valid acc: {:.3f}, test acc: {:.3f}\n'.format(acc_val, acc_test))
#             fout.write('acc of each exit: {}\n'.format(str(acc_each_stage.tolist())))
#             fout.write('test acc:{}\n'.format(acc_test))
#             if cnt == 18:
#                 acc70 = acc_test
#             if cnt == 7:
#                 acc50 = acc_test
#             cnt += 1
#     print('----------ALL DONE-----------')
#     return acc_test, acc70, acc50

def combine_tensor_across_processes(tnsr, dim):
    gathered = [torch.zeros(tnsr.shape, device=tnsr.device) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tnsr)
    combined_tensor = torch.cat(gathered, dim=dim)
    return combined_tensor

@torch.no_grad()
def exit_evaluate(model, train_dl, test_dl, output_dir, epoch, device, amp=True, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False,distributed=False):
    tester = Tester(model)
    model = model.cuda()
    val_pred, val_target = tester.calc_logit(train_dl, early_break=False, distributed=distributed)
    
    if distributed:
        val_pred = combine_tensor_across_processes(val_pred, dim=1)
        val_target = combine_tensor_across_processes(val_target, dim=0)
    test_pred, test_target = tester.calc_logit(test_dl, early_break=False, distributed=distributed)
    
    if distributed:
        test_pred = combine_tensor_across_processes(test_pred, dim=1)
        test_target = combine_tensor_across_processes(test_target, dim=0)
    each_exit = False
    
    dir_pth = output_dir / "exit_detail_slr"
    if not os.path.exists(dir_pth) and utils.is_main_process():
        os.makedirs(dir_pth)
        
    txt_pth = dir_pth / f"evaluate_repvit_fullfintune_epoch_{epoch}.txt"
    probs_list = [torch.tensor([0.4210, 0.3125, 0.0859, 0.1805]).to(device),
                    torch.tensor([0.1964, 0.2542, 0.1338, 0.4155]).to(device),
                    torch.tensor([0.0, 0.0, 0.0, 1.0]).to(device)]
    cnt = 0
    acc70 = 0.0
    acc50 = 0.0
    for probs in probs_list:
        # print('\n*****************')
        # print(probs)
        acc_val,  T = tester.dynamic_eval_find_threshold(val_pred, val_target, probs)
        # print(T)
        # print(f"------------rank{dist.get_rank()}'s acc_val: {acc_val}\n")
        acc_test, acc_each_stage = tester.dynamic_eval_with_threshold(test_pred, test_target, T)
        # print(f"------------rank{dist.get_rank()}'s acc_test: {acc_test}\n")
        # exit()
        print('valid acc: {:.3f}, test acc: {:.3f}'.format(acc_val, acc_test))
        print('acc of each exit: {}'.format(acc_each_stage))
        if utils.is_main_process():
            with open(txt_pth, 'w') as fout:
                fout.write('\n*****************\n')
                fout.write(str(probs.tolist()))
                fout.write('\n')
                fout.write(str(T.tolist()))
                fout.write('\n')
                fout.write('valid acc: {:.3f}, test acc: {:.3f}\n'.format(acc_val, acc_test))
                fout.write('acc of each exit: {}\n'.format(str(acc_each_stage.tolist())))
                fout.write('test acc:{}\n'.format(acc_test))
        if cnt == 1:
            acc70 = acc_test
        if cnt == 0:
            acc50 = acc_test
        cnt += 1
    print('----------ALL DONE-----------')
    return acc_test, acc70, acc50

def generate_distribution(each_exit=False):
    probs_list = []
    if each_exit:
        for i in range(4):
            probs = torch.zeros(4, dtype=torch.float)
            probs[i] = 1
            probs_list.append(probs)
    else:
        p_list = torch.zeros(34)
        for i in range(17):
            p_list[i] = (i + 4) / 20
            p_list[33 - i] = 20 / (i + 4)

        k = [0.85, 1, 0.5, 1]
        for i in range(33):
            probs = torch.exp(torch.log(p_list[i]) * torch.range(1, 4))
            probs /= probs.sum()
            for j in range(3):
                probs[j] *= k[j]
                probs[j+1:4] = (1 - probs[0:j+1].sum()) * probs[j+1:4] / probs[j+1:4].sum()
            probs_list.append(probs)
    return probs_list


class Tester(object):
    def __init__(self, model):
        # self.args = args
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()

    def calc_logit(self, dataloader, early_break=False, distributed=False):
        self.model.eval()
        n_stage = 4
        logits = [[] for _ in range(n_stage)]
        targets = []
        for i, (input, target) in enumerate(dataloader):
            if early_break and i > 100:
                break
            targets.append(target)
            input = input.cuda()
            with torch.no_grad():
                y1, y2, y3, y4 = self.model(input)
                output = [y1, y2, y3, y4]
                for b in range(n_stage):
                    _t = self.softmax(output[b])

                    logits[b].append(_t)
            if i % 50 == 0:
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        for b in range(n_stage):
            logits[b] = torch.cat(logits[b], dim=0)           # bs*len(dataloader)

        size = (n_stage, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_stage):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)
        
        if distributed:
            device = torch.device(f'cuda:{dist.get_rank()}')
            ts_logits = ts_logits.to(device)
            ts_targets = ts_targets.to(device)
        
        return ts_logits, ts_targets

    def dynamic_eval_find_threshold(self, logits, targets, p):
        """
            logits: m * n * c
            m: Stages
            n: Samples
            c: Classes
        """
        n_stage, n_sample, c = logits.size()

        max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

        _, sorted_idx = max_preds.sort(dim=1, descending=True)

        filtered = torch.zeros(n_sample)
        T = torch.Tensor(n_stage).fill_(1e8)

        for k in range(n_stage - 1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k])
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx]
                        break
            filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

        T[n_stage -1] = -1e8 # accept all of the samples at the last stage

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc = 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force the sample to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0
        for k in range(n_stage):
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, T

    def dynamic_eval_with_threshold(self, logits, targets,  T):
        n_stage, n_sample, _ = logits.size()
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False) # take the max logits as confidence

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc = 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force to exit at k
                    _g = int(gold_label.item())
                    _pred = int(argmax_preds[k][i].item())
                    if _g == _pred:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all, sample_all = 0, 0
        for k in range(n_stage):
            sample_all += exp[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, acc_rec / exp