import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


class Contrast_loss_his(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07):
        super(Contrast_loss_his, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, features_2=None, labels_2=None, reliability=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_samples, n_views, ...].
            labels: ground truth of shape [bsz, n_samples].
            mask: contrastive mask of shape [bsz, n_samples, n_samples], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            features_2: historical features
            labels_2: corresponding labels
            reliability: logits_mask_score of shape [bsz, n_samples]
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 4:
            raise ValueError('`features` needs to be [bsz, n_samples, n_views, ...],'
                             'at least 4 dimensions are required')
        if len(features_2.shape) < 4:
            raise ValueError('`features` needs to be [bsz, n_samples, n_views, ...],'
                             'at least 4 dimensions are required')
        if len(features.shape) > 4:
            features = features.view(features.shape[0], features.shape[1], features.shape[2], -1)
        if len(features_2.shape) > 4:
            features_2 = features_2.view(features_2.shape[0], features_2.shape[1], features_2.shape[2], -1)

        n_samples = features.shape[1]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # [bsz, bsz]
            mask = torch.eye(n_samples, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(labels.shape[0], -1, 1)
            labels_2 = labels_2.contiguous().view(labels_2.shape[0], -1, 1)
            if labels.shape[1] != n_samples:
                raise ValueError('Num of labels does not match num of features')
            if labels_2.shape[1] != n_samples:
                raise ValueError('Num of labels does not match num of features')
            # [bsz, bsz]
            reliability_mask = reliability.unsqueeze(1).repeat(1, reliability.shape[1], 1)
            mask = torch.eq(labels, labels_2.transpose(1,2)).float().to(device)
        else:
            # [bsz, bsz]
            mask = mask.float().to(device)


        contrast_count = features_2.shape[2]
        contrast_feature = torch.cat(torch.unbind(features_2, dim=2), dim=1)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, :, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.transpose(1,2)),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=2, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(1, anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            2,
            torch.arange(n_samples * anchor_count).view(1, -1, 1).repeat(features.shape[0], 1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask * reliability_mask
        log_prob = logits - torch.log(exp_logits.sum(2, keepdim=True)+1e-20) + torch.log(reliability_mask+1e-20)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob ).sum(2) / mask.shape[2]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(features.shape[0], anchor_count, n_samples).mean()

        return loss
