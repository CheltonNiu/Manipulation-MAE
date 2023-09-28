import torch, os
import numpy as np
from models import models_mae_iml
import matplotlib.pyplot as plt

# device config
device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
gpus = [0, 1]


# define the utils
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def prepare_model(chkpt_path, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae_iml, arch)()
    # load model
    checkpoint = torch.load(chkpt_path, map_location=device)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print('Checkpoint loaded. Version:', chkpt_path, msg)
    return model


def test_model(model, dataset, data_loader):
    confusion_matrix = {'TP': [], 'TN': [], 'FP': [], 'FN': [], 'ACC': [], 'P': [], 'N': []}
    for i, (imgs, masks, _) in enumerate(data_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        loss, pred, _, patch_label, label_pred = model(imgs, masks, mask_ratio=0)
        total_patch_count = patch_label.size()[1]
        P = (patch_label == 1).sum()
        N = (patch_label == 0).sum()
        label_xor = torch.logical_xor(label_pred, patch_label).to(torch.int64)  # 整体判断正确与否情况,相异为1

        Hit = (label_xor == 0).sum()
        TP = ((torch.logical_and(label_pred, patch_label).to(torch.int64)) == 1).sum()
        TN = ((torch.logical_and(torch.logical_not(label_pred), torch.logical_not(patch_label)).to(
            torch.int64)) == 1).sum()
        FP = ((torch.logical_and(label_pred, torch.logical_not(patch_label)).to(torch.int64)) == 1).sum()
        FN = ((torch.logical_and(torch.logical_not(label_pred), patch_label).to(torch.int64)) == 1).sum()

        acc = Hit / total_patch_count
        confusion_matrix['ACC'].append(acc.item())
        confusion_matrix['TP'].append(TP.item())
        confusion_matrix['FN'].append(FN.item())
        confusion_matrix['TN'].append(TN.item())
        confusion_matrix['FP'].append(FP.item())
        confusion_matrix['P'].append(P.item())
        confusion_matrix['N'].append(N.item())
        print(
            "\r{}/{}, Hit:{}/{}, TP/P:{}/{}, FN/P:{}/{}, FP/N:{}/{}, TN/N:{}/{}".format(i + 1, len(
                dataset), Hit, total_patch_count, TP, P, FN, P, FP, N, TN, N), end='', flush=True)

    acc_mean = np.mean(confusion_matrix['ACC'])
    TP = np.sum(confusion_matrix['TP'])
    FN = np.sum(confusion_matrix['FN'])
    TN = np.sum(confusion_matrix['TN'])
    FP = np.sum(confusion_matrix['FP'])
    P = np.sum(confusion_matrix['P'])
    N = np.sum(confusion_matrix['N'])
    print("\nAccuracy mean:", acc_mean, "TP:", TP, "FN:", FN, "TN:", TN, "FP:", FP, "P:",
          P, "N:", N)
    recall = TP / P
    precision = TP / (TP + FP)
    F1score = 2 * precision * recall / (recall + precision)
    print('recall:{:.2%}, precision:{:.2%}, F1-score:{:.2%}'.format(recall, precision, F1score))


def visualize_model(model, dataset, data_loader):

    def show_image(image, title='', label=False):
        # image is [H, W, 3]
        assert image.shape[2] == 3
        if not label:
            plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
            plt.title(title)
            plt.axis('off')
        else:
            plt.imshow(image)
            plt.title(title)
            # plt.axis('off')
            plt.xticks([])
            plt.yticks([])
        return

    for i, (imgs, masks, paths) in enumerate(data_loader):
        imgs, masks, paths = imgs.to(device), masks.to(device), paths
        loss, pred, mask, patch_label, label_pred = model(imgs, masks, mask_ratio=0)
        TP = ((torch.logical_and(label_pred, patch_label).to(torch.int64)) == 1).sum()

        pred = model.unpatchify(pred)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()
        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        imgs = torch.einsum('nchw->nhwc', imgs).cpu()

        # label show
        patch_label_img = model.unpatchify(
            patch_label.detach().repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3))
        patch_label_img = torch.einsum('nchw->nhwc', patch_label_img).detach().cpu()

        label_pred_img = model.unpatchify(
            label_pred.detach().repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3))
        label_pred_img = torch.einsum('nchw->nhwc', label_pred_img).detach().cpu().float()

        imgs_masked = imgs * (1 - mask)
        imgs_paste = imgs * (1 - mask) + pred * mask

        # make the plt figure larger
        # plt.rcParams['figure.figsize'] = [50, 50]

        plt.subplot(2, 3, 1)
        show_image(imgs[0], "original")

        plt.subplot(2, 3, 2)
        show_image(imgs_masked[0], "masked")

        plt.subplot(2, 3, 3)
        show_image(pred[0], "reconstruction")

        plt.subplot(2, 3, 4)
        show_image(imgs_paste[0], "reconstruction + visible")

        plt.subplot(2, 3, 5)
        show_image(patch_label_img[0], "groundtruth label", label=True)

        plt.subplot(2, 3, 6)
        show_image(label_pred_img[0], "predict label", label=True)

        # plt.show()
        if TP > 0:
            plt.savefig(os.path.join('./results', os.path.basename(paths[0])))
            print(paths[0])
        plt.close("all")
