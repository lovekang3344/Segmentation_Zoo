import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1e5

        # 展平
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.mean()
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()



    def forward(self, input, targets):
        """
            input tensor of shape = (N, C, H, W)
            targets tensor of shape = (N, H, W)
        """

        # 这里先使用 one_hot 将 target 转为 (N, C, H, W)
        nclass = input.shape[1]
        targets = (F.one_hot(targets.long(), nclass)).squeeze(1).permute(0, 3, 1, 2)

        assert input.shape == targets.shape, "predict & target shape do not match"

        binaryDiceLoss = BinaryDiceLoss()
        total_loss = 0

        # 归一化输出
        logits = F.softmax(input, dim=1)

        # 遍历所有的channels
        channels = targets.shape[1]

        for i in range(channels):
            dice_loss = binaryDiceLoss(logits[:, i], targets[:, i])
            total_loss += dice_loss

        return total_loss / channels

def iou(pred, target, n_classes = 21):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # 忽略背景
    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        # pred_inds = pred == cls
        # target_inds = target == cls
        # intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        # union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        intersection = torch.sum((pred == cls) & (target == cls))
        union = torch.sum((pred == cls) | (target == cls))
        if union == 0:
            # ious.append(0)  # If there is no ground truth, do not include in evaluation
            continue
        else:
            ious.append(float(intersection) / float(union))
    return np.mean(ious)




def train(trainloader, testloader, model, epoches_num, criterion, optim, device, save_path):
    writer = SummaryWriter('./runs/Adam')
    mx_Accuracy = 0
    for epoch in range(epoches_num):
        loss_record = []
        acc_record = []
        step = 0
        model.train()
        for x, y in trainloader:
            optim.zero_grad()
            x, y = x.to(device), y.to(device)

            predict = model(x)
            loss = criterion(predict, y.squeeze(1).long())
            # loss = criterion(predict, y)


            predict = predict.permute(0, 2, 3, 1)
            predict = torch.argmax(predict, dim=-1)
            acc = iou(predict, y, 21)
            loss_record.append(loss.item())
            acc_record.append(acc)

            loss.backward()

            optim.step()

        train_loss = sum(loss_record) / len(loss_record)
        train_acc = sum(acc_record) / len(acc_record)

        loss_record = []
        acc_record = []
        model.eval()
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y.squeeze(1).long())
                pred = pred.permute(0, 2, 3, 1)
                pred = torch.argmax(pred, dim=-1)
                acc = iou(pred, y, 21)

            loss_record.append(loss.item())
            acc_record.append(acc)

        val_loss = sum(loss_record) / len(loss_record)
        val_acc = sum(acc_record) / len(acc_record)


        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Train Miou', train_acc, epoch)
        writer.add_scalar('validation Miou', val_acc, epoch)
        print(f"[{epoch+1}/{epoches_num}]\ttrain_Loss:  {train_loss}\tval_loss:  {val_loss}")
        print(f"[{epoch+1}/{epoches_num}]\ttrain_ACC:  {train_acc}\tval_ACC:  {val_acc}")
        if mx_Accuracy < val_acc:
            mx_Accuracy = val_acc
            torch.save(model.state_dict(), save_path+'/best_model.pth')
        if epoch + 1 == epoches_num:
            torch.save(model.state_dict(), 'save_model/SegNet/last_model.pth')

    print(f"最佳的Accuracy：{mx_Accuracy}")
