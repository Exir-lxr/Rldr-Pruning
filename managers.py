# Author: Xiaoru Liu (Xavier Liu)
# Github: https://github.com/Exir-lxr/RldrInPruning.git
# Email: lxr_orz@126.com

from .statistic import StatisticManager
from .forward_map import ForwardMapManger
import torch
import numpy as np


class RIPManager(object):

    def __init__(self, using_statistic=True, structured_mode=False,
                 efficiency_index='channel', efficiency_index_mapping=lambda x: x):

        if efficiency_index in ['channel', 'para', 'flops']:
            self.efficiency_index = efficiency_index
        else:
            raise Exception('efficiency_index should be channel or para or flops. '
                            'Now is:', efficiency_index)

        self.efficiency_index_mapping = efficiency_index_mapping

        self.name_to_statistic = {}
        self.name_to_info = {}
        self.name_module = {}

        # For structured_mode, the final result dont need any additional mask
        self.structured_mode = structured_mode

        # If prune result needs additional mask, RIPManager is used to add these mask.
        # Else, RIPManager is used to reshape the networks
        self.statistic = using_statistic

        self.map_manager = None

    # CALL After checkpoint(no mask) loaded and Before model putted into GPU
    # After CALL, load checkpoint(with mask)
    def __call__(self, model, inputs_size):

        self.map_manager = ForwardMapManger(model)
        self.map_manager.collect_info(inputs_size)

        for name, sub_module in model.named_modules():

            if isinstance(sub_module, torch.nn.Conv2d) and sub_module.groups == 1:
                if sub_module.kernel_size[0] == 1:

                    self.name_module[name] = sub_module
                    info = StatisticManager(sub_module, statistic_mode=self.statistic)
                    self.name_to_statistic[name] = info
                    self.name_to_info[name] = info
                    info.module.register_buffer('output_shrink_mask',
                                                torch.ones(info.out_channel_num, dtype=torch.float))
                else:
                    self.name_module[name] = sub_module
                    info = StatisticManager(sub_module, statistic_mode=False)
                    self.name_to_info[name] = info
                    info.module.register_buffer('output_shrink_mask',
                                                torch.ones(info.out_channel_num, dtype=torch.float))

            elif isinstance(sub_module, torch.nn.Linear):

                self.name_module[name] = sub_module
                info = StatisticManager(sub_module, statistic_mode=self.statistic)
                self.name_to_statistic[name] = info
                self.name_to_info[name] = info
                info.module.register_buffer('output_shrink_mask',
                                            torch.ones(info.out_channel_num, dtype=torch.float))
            # else:
            #     i = 0
            #     for _ in sub_module.children():
            #         i += 1
            #         break
            #     if i == 0:
            #        print('unmanagered module:', sub_module, 'named:', self.map_manager.id_name_dict[id(sub_module)])

        for name in self.name_to_statistic:

            info = self.name_to_statistic[name]
            conv_module = self.map_manager.name_pointer[name]
            bn_module = self.map_manager.get_corresponding_bn(conv_module)

            if bn_module is not None:
                info.bn_module = bn_module
            elif info.module.bias is None:
                print('No bias or bn found in ', info.module)

        self._update_out_prune_mask()

    def _computer_statistic(self):

        print('Computing statistics...')

        with torch.no_grad():

            for name in self.name_to_statistic:

                info = self.name_to_statistic[name]

                info.compute_statistic()

                info.clear_zero_variance()

                info.compute_score()

    def prune_once(self):
        if self.structured_mode:
            return self._structure_prune_once()
        else:
            return self._normal_prune_once()

    def _normal_prune_once(self):
        name_score_mask_dict = {}

        for name in self.name_to_statistic:
            info = self.name_to_statistic[name]
            score, mask = info.get_score_mask()
            name_score_mask_dict[name] = [score * mask, mask]

        min_score = 1000
        the_name = None
        index = None
        idx = None

        for name in name_score_mask_dict:

            score, mask = name_score_mask_dict[name]
            sorted_index = np.argsort(score)
            in_flag = False

            for idx in list(sorted_index):
                idx = int(idx)
                if int(mask[idx]) != 0:
                    in_flag = True
                    break

            if in_flag:
                ss = float(score[idx])
                if ss < min_score:
                    min_score = ss
                    the_name = name
                    index = idx
            else:
                raise Exception('ALL of the input tensors are pruned in some module!')

        info = self.name_to_statistic[the_name]
        print('pruned score: ', min_score, 'name: ', the_name)
        info.prune_then_modify(index)
        info.compute_score()
        if min_score == 0.0:
            return 0
        else:
            self._update_out_prune_mask(self.name_module[the_name])
            return 1

    def _structure_prune_once(self):
        name_score_mask_dict = {}

        for name in self.name_to_statistic:
            info = self.name_to_statistic[name]
            score, mask = info.get_score_mask()
            delta_para, delta_flops = self._para_flops_drop(name)
            if self.efficiency_index == 'para':
                mod_score = score / self.efficiency_index_mapping(delta_para)
            elif self.efficiency_index == 'flops':
                mod_score = score / self.efficiency_index_mapping(delta_flops)
            else:
                mod_score = score
            name_score_mask_dict[name] = [mod_score * mask, mask, score*mask]

        module_groups, id_group_index = self.map_manager.get_modules_with_same_input()

        # average score
        for a_group in module_groups:

            total_num = len(a_group)
            total_score = None
            for a_module in a_group:
                the_name = self.map_manager.id_name_dict[id(a_module)]
                if total_score is None:
                    total_score = name_score_mask_dict[the_name][0]
                else:
                    total_score += name_score_mask_dict[the_name][0]
            new_score = total_score / total_num
            for a_module in a_group:
                the_name = self.map_manager.id_name_dict[id(a_module)]
                name_score_mask_dict[the_name][0] = new_score

        min_score = 1000
        the_name = None
        index = None
        idx = None

        for name in name_score_mask_dict:

            score, mask, pure_score = name_score_mask_dict[name]
            sorted_index = np.argsort(score)
            in_flag = False

            for idx in list(sorted_index):
                idx = int(idx)
                if int(mask[idx]) != 0:
                    in_flag = True
                    break

            if in_flag:
                ss = float(score[idx])
                if ss < min_score:
                    min_score = ss
                    the_name = name
                    index = idx
            else:
                raise Exception('ALL of the input tensors are pruned in some module!')

        conv_module = self.map_manager.name_pointer[the_name]
        if id(conv_module) in id_group_index:
            the_group_index = id_group_index[id(conv_module)]
            print('together')
            the_name_list = [self.map_manager.id_name_dict[id(tar_module)]
                             for tar_module in module_groups[the_group_index]]
        else:
            the_name_list = [the_name]

        for name in the_name_list:
            info = self.name_to_statistic[name]
            print('pruned score: ', min_score, 'pure score: ',
                  name_score_mask_dict[name][2][index], 'name: ', name)
            info.prune_then_modify(index)
            info.compute_score()
        if min_score == 0.0:
            return 0
        else:
            self._update_out_prune_mask(self.name_module[the_name_list[0]])
            return len(the_name_list)

    # Load mask before this function
    def prune(self, pruned_num):
        self._update_out_prune_mask()
        self._computer_statistic()

        ii = 0

        while ii < pruned_num:

            it_pruned_num = self.prune_once()
            ii += it_pruned_num

        for name in self.name_to_statistic:

            info = self.name_to_statistic[name]

            info.de_normalize_bn()

    def shrink_afterward_lines(self):
        # It is about Markov Cover

        self._update_out_prune_mask()

        for name in self.name_to_info:

            info = self.name_to_info[name]
            final_output_shrink_mask = torch.sign(info.module.output_shrink_mask)

            son_node = self.map_manager.get_next_channel_change_modules(info.module)

            father_node = []
            for a_son in son_node:
                for a_father in self.map_manager.get_previous_channel_change_modules(a_son):
                    if a_father not in father_node:
                        father_node.append(a_father)

            for a_father in father_node:
                self.name_to_info[self.map_manager.id_name_dict[id(a_father)]].\
                    shrink(output_shrink_mask=final_output_shrink_mask)
                for trivial_module in self.map_manager.get_channel_straight_line(a_father):
                    self._shrink_for_channel_straight_module(trivial_module, final_output_shrink_mask)

            for a_after_module in son_node:
                the_name = self.map_manager.id_name_dict[id(a_after_module)]
                the_info = self.name_to_statistic[the_name]
                the_info.shrink(input_shrink_mask=final_output_shrink_mask)

        print('Remove statistic hooks')
        self.remove_statistic_hook()

    def _update_out_prune_mask(self, specific_module=None):
        if specific_module is not None:
            father_node = self.map_manager.get_previous_channel_change_modules(specific_module)
            for a_father_node in father_node:
                son_node = self.map_manager.get_next_channel_change_modules(a_father_node)
                if len(son_node) > 0:
                    a_father_node.output_shrink_mask.zero_()
                    for a_son in son_node:
                        a_father_node.output_shrink_mask.data = a_father_node.output_shrink_mask + a_son.mask

        else:
            for name in self.name_to_info:
                info = self.name_to_info[name]
                son_node = self.map_manager.get_next_channel_change_modules(info.module)
                if len(son_node) > 0:
                    info.module.output_shrink_mask.zero_()
                    for a_son in son_node:
                        info.module.output_shrink_mask.data = info.module.output_shrink_mask + a_son.mask

    def _para_flops_drop(self, name):
        if self.structured_mode:
            para = 0
            flops = 0
            handled_node = []
            info = self.name_to_info[name]

            father_node = self.map_manager.get_previous_channel_change_modules(info.module)

            _para, _flops = self.map_manager.remove_one_channel(
                info.module, info.module.mask, info.module.output_shrink_mask, True)
            para += _para
            flops += _flops
            handled_node.append(info.module)

            for a_father in father_node:

                _para, _flops = self.map_manager.remove_one_channel(
                    a_father, a_father.mask, a_father.output_shrink_mask, False)
                para += _para
                flops += _flops

                for trivial_module in self.map_manager.get_channel_straight_line(a_father):
                    _para, _flops = self.map_manager.remove_one_channel(
                        trivial_module, a_father.output_shrink_mask, a_father.output_shrink_mask, True)
                    para += _para
                    flops += _flops
                son_node = self.map_manager.get_next_channel_change_modules(a_father)
                for a_after_module in son_node:
                    if a_after_module not in handled_node:
                        _para, _flops = self.map_manager.remove_one_channel(
                            a_after_module, a_after_module.mask, a_after_module.output_shrink_mask, True)
                        para += _para
                        flops += _flops
                        handled_node.append(a_after_module)
        else:
            raise Exception('Not support yet.')
        return para*1.0/len(handled_node), flops*1.0/len(handled_node)

    def pruning_overview(self):

        all_channel_num = 0
        remained_channel_num = 0

        for name in self.name_to_info:

            info = self.name_to_info[name]
            r, a = info.query_channel_num()
            all_channel_num += a
            remained_channel_num += r
            print(name, r, '/', a)

        print('channel number: ', remained_channel_num, '/', all_channel_num)
        para, flops = self.map_manager.get_para_flops()
        mpara = round(para * 1e-6, 2)
        mflops = round(flops * 1e-6, 2)
        print('Total para:', mpara, 'M  || Total flops:', mflops, 'M')
        return remained_channel_num, all_channel_num, para, flops

    def reset_statistic(self):

        for name in self.name_to_statistic:

            info = self.name_to_statistic[name]

            info.reset_statistic()

    def remove_statistic_hook(self):

        for name in self.name_to_statistic:
            if self.name_to_statistic[name].is_statistic_mode:
                self.name_to_statistic[name].remove_statistic_hooks()

    def remove_mask_hook(self):

        for name in self.name_to_statistic:
            self.name_to_statistic[name].remove_mask_hook()

    def visualize(self):

        from matplotlib import pyplot as plt
        i = 1
        for name in self.name_to_statistic:
            info = self.name_to_statistic[name]
            forward_mean = info.f_cls.sum_mean / info.f_cls.counter
            forward_cov = (info.f_cls.sum_covariance / info.f_cls.counter) - \
                torch.mm(forward_mean.view(-1, 1), forward_mean.view(1, -1))

            grad_mean = info.b_cls.sum_mean / info.b_cls.counter
            grad_cov = (info.b_cls.sum_covariance / info.b_cls.counter) - \
                torch.mm(grad_mean.view(-1, 1), grad_mean.view(1, -1))
            plt.subplot(10, 15, i)
            plt.imshow(np.array(forward_cov.cpu()), cmap='hot')
            plt.xticks([])
            plt.yticks([])
            i += 1
            plt.subplot(10, 15, i)
            plt.imshow(np.array(grad_cov.cpu()), cmap='hot')
            plt.xticks([])
            plt.yticks([])
            i += 1
            if i > 150:
                print('Only part of them are shown')
                break
        plt.show()

    @staticmethod
    def _shrink_for_channel_straight_module(module, shrink_mask):

        if isinstance(module, torch.nn.BatchNorm2d):
            if shrink_mask.shape[0] == module.weight.shape[0]:
                shrink_indexes = shrink_mask.to_sparse().indices().view([-1])
                module.num_features = len(shrink_indexes)
                module.weight.data = torch.index_select(module.weight, 0, shrink_indexes)
                module.bias.data = torch.index_select(module.bias, 0, shrink_indexes)
                module.running_mean.data = torch.index_select(module.running_mean, 0, shrink_indexes)
                module.running_var.data = torch.index_select(module.running_var, 0, shrink_indexes)
            else:
                print('SKIP:', module, 'with mask:', shrink_mask.shape)
        elif isinstance(module, torch.nn.Conv2d):
            if shrink_mask.shape[0] == module.weight.shape[0]:
                shrink_indexes = shrink_mask.to_sparse().indices().view([-1])
                remain_num = len(shrink_indexes)
                module.in_channels = remain_num
                module.groups = remain_num
                module.out_channels = remain_num
                module.weight.data = torch.index_select(module.weight, 0, shrink_indexes)
                if module.bias is not None:
                    module.bias.data = torch.index_select(module.bias, 0, shrink_indexes)
            else:
                print('SKIP:', module, 'with mask:', shrink_mask.shape)
