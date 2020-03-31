# Author: Xiaoru Liu (Xavier Liu)
# Github: https://github.com/Exir-lxr/RldrInPruning.git
# Email: lxr_orz@126.com


import torch
import numpy as np


def compute_statistic_and_update(samples, sum_mean, sum_covar, counter) -> None:
    samples = samples.to(torch.half).to(torch.double)
    samples_num = list(samples.shape)[0]
    counter += samples_num
    sum_mean += torch.sum(samples, dim=0)
    sum_covar += torch.mm(samples.permute(1, 0), samples)


class ForwardStatisticHook(object):

    def __init__(self, module):
        self.module = module

        if isinstance(module, torch.nn.Conv2d):
            channel_num = module.in_channels
        elif isinstance(module, torch.nn.Linear):
            channel_num = module.in_features
        else:
            raise Exception('Unsupported module')

        self.channel_num = channel_num

        module.register_buffer('sum_mean', torch.zeros(channel_num, dtype=torch.double))
        module.register_buffer('sum_covariance', torch.zeros([channel_num, channel_num], dtype=torch.double))
        module.register_buffer('counter', torch.zeros(1, dtype=torch.double))

    def hook(self, module, inputs, output) -> None:
        with torch.no_grad():
            channel_num = self.channel_num
            # from [N,C,W,H] to [N*W*H,C]
            if isinstance(module, torch.nn.Conv2d):
                samples = inputs[0].permute(0, 2, 3, 1).contiguous().view(-1, channel_num)
            elif isinstance(module, torch.nn.Linear):
                samples = inputs[0]
            compute_statistic_and_update(samples, module.sum_mean, module.sum_covariance, module.counter)

    def reset(self):
        self.module.sum_mean.zero_()
        self.module.sum_covariance.zero_()
        self.module.counter.zero_()

    def remove_buffer(self):
        del self.module.sum_mean
        del self.module.sum_covariance
        del self.module.counter


class MaskHook(object):

    def __init__(self, module):
        self.module = module
        self.is_real_prune = False
        if isinstance(module, torch.nn.Conv2d):
            self.channel_num = module.in_channels
        elif isinstance(module, torch.nn.Linear):
            self.channel_num = module.in_features
        module.register_buffer('mask', torch.ones(self.channel_num))

    def hook(self, module, inputs):
        if self.is_real_prune:
            indexes = self.module.mask.to_sparse().indices().view([-1])
            if isinstance(module, torch.nn.Conv2d):
                modified = torch.index_select(inputs[0], 1, indexes)
                return modified
            elif isinstance(module, torch.nn.Linear):
                return torch.index_select(inputs[0], 1, indexes)
        else:
            if isinstance(module, torch.nn.Conv2d):
                modified = torch.mul(inputs[0].permute([0, 2, 3, 1]), module.mask)
                return modified.permute([0, 3, 1, 2])
            elif isinstance(module, torch.nn.Linear):
                return torch.mul(inputs[0], module.mask)

    def remove_buffer(self):
        del self.module.mask


class StatisticManager(object):

    def __init__(self, module, statistic_mode=False):

        self.is_statistic_mode = statistic_mode

        # init
        self.module = module

        if isinstance(module, torch.nn.Conv2d):
            self.out_channel_num = module.out_channels
        elif isinstance(module, torch.nn.Linear):
            self.out_channel_num = module.out_features
        else:
            raise Exception('Unsupported module.', module)

        self.pre_f_cls = MaskHook(module)
        self.mask_hook = module.register_forward_pre_hook(self.pre_f_cls.hook)

        if statistic_mode:
            self.statistic_hooks = []

            self.f_cls = ForwardStatisticHook(module)
            self.statistic_hooks.append(module.register_forward_hook(self.f_cls.hook))

            # the inputs channel number
            self.in_channel_num = self.pre_f_cls.channel_num

            # forward statistic
            self.forward_mean = None
            self.variance = None
            self.forward_cov = None

            # forward info
            self.zero_variance_masked_zero = None
            self.zero_variance_masked_one = None
            self.de_correlation_variance = None

            # raw score for rldr-pruning
            self.alpha = None
            self.normalized_alpha = None
            self.stack_op_for_weight = None

            # weights
            self.weight = None

            # corresponding bn layer
            self.bn_module = None

    def compute_statistic(self, threshold=10e-5):
        # compute forward statistic
        self.forward_mean = self.f_cls.module.sum_mean / self.f_cls.module.counter
        self.forward_cov = (self.f_cls.module.sum_covariance / self.f_cls.module.counter) - \
            torch.mm(self.forward_mean.view(-1, 1), self.forward_mean.view(1, -1))

        # equal 0 where variance of an activate is 0
        self.variance = torch.diag(self.forward_cov)
        self.zero_variance_masked_zero = torch.ge(torch.abs(self.variance), threshold).to(torch.double)
        # self.zero_variance_masked_zero = torch.sign(self.variance)

        # where 0 var compensate 1
        self.zero_variance_masked_one = - self.zero_variance_masked_zero + 1

        self._normalize_bn()

    def _normalize_bn(self):
        # According to statistic, let EY and DY equal E(Y) and D(Y) then modifying scale and shift
        if self.bn_module is not None:
            weight = torch.squeeze(self.module.weight)
            self.bn_module.register_buffer('weight_tmp', self.bn_module.weight.data)
            self.bn_module.register_buffer('bias_tmp', self.bn_module.bias.data)
            ey = torch.squeeze(torch.mm(weight, self.forward_mean.to(torch.float).view(-1, 1)))
            dy = torch.diag(torch.mm(torch.mm(weight, self.forward_cov.to(torch.float)), weight.t()))
            new_weight = self.bn_module.weight * torch.sqrt(dy/self.bn_module.running_var)
            new_bias = self.bn_module.bias + \
                self.bn_module.weight*(ey-self.bn_module.running_mean)/torch.sqrt(self.bn_module.running_var)
            self.bn_module.weight.data = new_weight
            self.bn_module.bias.data = new_bias
            self.bn_module.running_mean.data = ey
            self.bn_module.running_var.data = dy

    def clear_zero_variance(self):

        # according to [zero variance mask], find all the channels with 0 variance,
        # then set corresponding [weights] to 0,
        # and update parameters in [bn module] or [biases]

        verify = int(torch.sum(self.module.mask
                               - self.zero_variance_masked_zero.to(torch.float)
                               * self.module.mask))
        tmp_verify = int(torch.sum(self.module.mask - self.zero_variance_masked_zero.to(torch.float)))

        if verify != tmp_verify:
            raise Exception('mask zero but variance not zero.')

        if verify != 0:

            print('Number of more zero variance channel: ', verify)

            # update weight
            if len(self.module.weight.shape) == 4:
                self.module.weight.data[:, :, 0, 0] = \
                    torch.squeeze(self.module.weight) * self.zero_variance_masked_zero.to(torch.float)
            elif len(self.module.weight.shape) == 2:
                self.module.weight.data[:, :] = self.module.weight * self.zero_variance_masked_zero.to(torch.float)

            # update bn or biases
            if self.bn_module is None:
                delta_weight = torch.squeeze(self.module.weight) * self.zero_variance_masked_one.to(torch.float)
                self.module.bias.data -= \
                    torch.squeeze(torch.mm(delta_weight, self.forward_mean.to(torch.float).view(-1, 1)))
            else:
                self.bn_module.running_mean.data = \
                    torch.squeeze(torch.mm(torch.squeeze(self.module.weight),
                                           self.forward_mean.to(torch.float).view(-1, 1)))

    def compute_score(self):
        # for a group of given [self.forward_mean, self.forward_cov, self.grad_mean, self.grad_cov]
        # compute all kinds of RIP score
        repaired_forward_cov = self.forward_cov + torch.diag(self.zero_variance_masked_one)

        f_cov_inverse = repaired_forward_cov.inverse()

        repaired_alpha = torch.reciprocal(torch.diag(f_cov_inverse))
        self.de_correlation_variance = repaired_alpha * self.zero_variance_masked_zero

        self.stack_op_for_weight = (f_cov_inverse.t() * repaired_alpha.view(1, -1)).t()

        # fetch weights
        self.weight = self.module.weight.detach()

        # score for rldr-pruning
        self.alpha = \
            torch.sum(torch.pow(torch.squeeze(self.weight), 2), dim=0).to(torch.double) * self.de_correlation_variance
        self.normalized_alpha = self.alpha / torch.norm(self.alpha)

        return self.normalized_alpha

    def prune_then_modify(self, index_of_channel):
        # A global manager gives the prune order.
        # Therefore, update [mask], then [weights],

        channel_mask = self.module.mask
        if channel_mask[index_of_channel] == 0:
            print('Already pruned.')
        else:
            # update [mask]
            channel_mask[index_of_channel] = 0
            self.module.mask.data = channel_mask

            # update [weights]

            delta_weight = torch.mm(self.weight[:, index_of_channel].view(-1, 1).to(torch.double),
                                    self.stack_op_for_weight[index_of_channel, :].view(1, -1)).to(torch.float)

            new_weight = torch.squeeze(self.weight) - delta_weight

            if isinstance(self.module, torch.nn.Conv2d):
                self.weight[:, :, 0, 0] = new_weight
            elif isinstance(self.module, torch.nn.Linear):
                self.weight[:, :] = new_weight
            else:
                raise Exception('Unsupported module')
            self.module.weight.data = self.weight

            # update [bn]

            if self.bn_module is None:
                print('Modify biases in', self.module)
                # On mobilenetv2 classify network, this only appear in the last classifier layer.
                # Exp shows that no changing in last classifier layer achieves better result.
                self.module.bias.data -= \
                    torch.squeeze(torch.mm(delta_weight, self.forward_mean.to(torch.float).view(-1, 1)))
            else:
                self.bn_module.running_mean.data = \
                    torch.squeeze(torch.mm(new_weight, self.forward_mean.to(torch.float).view(-1, 1)))
                self.bn_module.running_var.data = \
                    torch.diag(torch.mm(torch.mm(new_weight, self.forward_cov.to(torch.float)), new_weight.t()))

            # update statistic
            self.forward_cov[:, index_of_channel] = 0
            self.forward_cov[index_of_channel, :] = 0
            self.forward_mean[index_of_channel] = 0

            self.zero_variance_masked_zero = self.zero_variance_masked_zero * channel_mask.to(torch.double)
            self.zero_variance_masked_one = (1 - self.zero_variance_masked_zero).to(torch.double)

    def de_normalize_bn(self):
        # For a easier fine-tuning.
        if self.bn_module is not None:

            new_dy = self.bn_module.running_var * torch.pow(self.bn_module.weight_tmp / self.bn_module.weight, 2)
            new_ey = self.bn_module.running_mean - (self.bn_module.bias - self.bn_module.bias_tmp) * new_dy
            new_weight = self.bn_module.weight * torch.sqrt(new_dy / self.bn_module.running_var)
            new_bias = self.bn_module.bias + self.bn_module.weight * \
                (new_ey - self.bn_module.running_mean) / torch.sqrt(self.bn_module.running_var)
            self.bn_module.weight.data = new_weight
            self.bn_module.bias.data = new_bias
            self.bn_module.running_mean.data = new_ey
            self.bn_module.running_var.data = new_dy

            del self.bn_module.bias_tmp
            del self.bn_module.weight_tmp

    def get_score_mask(self):
        score = np.array(self.normalized_alpha.cpu())

        channel_mask = self.module.mask

        return score, np.array(channel_mask.cpu())

    def reset_statistic(self):
        self.f_cls.reset()

    def remove_statistic_hooks(self):

        for a_hook in self.statistic_hooks:

            a_hook.remove()

    def query_channel_num(self):

        channel_mask = self.module.mask

        return int(torch.sum(channel_mask).cpu()), int(channel_mask.shape[0])

    def shrink(self, input_shrink_mask=None, output_shrink_mask=None):
        if input_shrink_mask is not None:
            if input_shrink_mask.shape[0] == self.module.mask.shape[0]:
                # print(input_shrink_mask, self.module.weight.shape)
                input_indexes = input_shrink_mask.to_sparse().indices().view([-1])
                self.module.mask.data = torch.index_select(self.module.mask, 0, input_indexes)
                self.module.weight.data = torch.index_select(self.module.weight, 1, input_indexes)
            else:
                print('SKIP:', self.module, 'with mask:', input_shrink_mask.shape)
        if output_shrink_mask is not None:
            if output_shrink_mask.shape[0] == self.module.weight.shape[0]:
                # print(output_shrink_mask, self.module.weight.shape)
                output_indexes = output_shrink_mask.to_sparse().indices().view([-1])
                self.module.weight.data = torch.index_select(self.module.weight, 0, output_indexes)
                if self.module.bias is not None:
                    self.module.bias.data = torch.index_select(self.module.bias, 0, output_indexes)
            else:
                print('SKIP:', self.module, 'with mask:', output_shrink_mask.shape)

    def remove_mask_hook(self):
        del self.module.mask
        self.mask_hook.remove()
