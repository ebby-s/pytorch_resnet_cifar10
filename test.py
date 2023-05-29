import argparse
import time
import glob

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

from mpemu import mpt_emu

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')



def main():
    global args
    args = parser.parse_args()

    # Setup stuff.
    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()

    test_archs = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

    frmts_torch_fp = ['fp32', 'fp16', 'bf16']

    # Test a pretrained model for several formats.
    print("----------------- Testing model " + args.arch)

    pretrained_model = glob.glob('pretrained_models/' + args.arch + '-*')[0]

    for frmt in ['int32']:  # Test fp formats from torch.

        print("Testing format " + frmt)

        model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
        model.cuda()
        model.load_state_dict(torch.load(pretrained_model)['state_dict'])

        if(frmt == 'int32'):
            model.qint32()
            criterion.qint32()

        validate(val_loader, model, criterion, frmt)

    for frmt in frmts_torch_fp:  # Test fp formats from torch.

        print("Testing format " + frmt)

        model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
        model.cuda()
        model.load_state_dict(torch.load(pretrained_model)['state_dict'])

        if(frmt == 'fp16'):
            model.half()
            criterion.half()
        elif(frmt == 'bf16'):
            model.bfloat16()
            criterion.bfloat16()

        validate(val_loader, model, criterion, frmt)

    for frmt in ['e3m4', 'e4m3', 'e5m2']:  # Test fp8 formats.

        print("Testing format " + frmt)

        model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
        model.cuda()
        model.load_state_dict(torch.load(pretrained_model)['state_dict'])

        model, emulator = mpt_emu.quantize_model(model, dtype=frmt)
        model = emulator.fuse_bnlayers_and_quantize_model(model)

        validate(val_loader, model, criterion)


def validate(val_loader, model, criterion, frmt=None):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if frmt == 'fp16':
                input_var = input_var.half()
            elif frmt == 'bf16':
                input_var = input_var.bfloat16()
            elif(frmt == 'int32'):
                input_var = input_var.qint32()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
