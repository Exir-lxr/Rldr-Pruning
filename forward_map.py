# Author: Xiaoru Liu (Xavier Liu)
# Github: https://github.com/Exir-lxr/RldrInPruning.git
# Email: lxr_orz@126.com

import torch
import numpy as np


class JointTensor(torch.Tensor):

    def __init__(self, tensor):
        torch.Tensor.__init__(self)
        torch.Tensor.__eq__(self, tensor)
        # {name: Module, ...}
        self.input_module_list = []

    def record_input_module(self, module):
        self.input_module_list.append(module)

    def view(self, size):
        result = super(JointTensor, self).view(size)
        out = JointTensor(result)
        out.input_module_list = self.input_module_list
        return out

    def __add__(self, other):
        if isinstance(other, JointTensor):
            result = super(JointTensor, self).__add__(other)
            out = JointTensor(result)
            out.input_module_list = self.input_module_list + other.input_module_list
        elif isinstance(other, torch.Tensor):
            result = super(JointTensor, self).__add__(other)
            out = JointTensor(result)
            out.input_module_list = self.input_module_list
        else:
            raise Exception('JointTensor add JointTensor or Tensor')
        return out

    def __mul__(self, other):
        if isinstance(other, JointTensor):
            result = super(JointTensor, self).__mul__(other)
            out = JointTensor(result)
            out.input_module_list = self.input_module_list + other.input_module_list
        elif isinstance(other, torch.Tensor):
            result = super(JointTensor, self).__mul__(other)
            out = JointTensor(result)
            out.input_module_list = self.input_module_list
        else:
            raise Exception('JointTensor add JointTensor or Tensor')
        return out

    def mean(self, *args):
        result = super(JointTensor, self).mean(*args)
        out = JointTensor(result)
        out.input_module_list = self.input_module_list

        return out

    def echo_list(self):
        return self.input_module_list

    def reset_list(self):
        self.input_module_list = []

    def view(self, *size):
        t = super().view(*size)
        new = JointTensor(t)
        new.input_module_list = self.input_module_list
        return new


class ForwardMapManger(object):

    def __init__(self, model: torch.nn.Module):

        self.hooks = []
        self.module = model

        self.head = []
        self.tail = []

        self.name_pointer = {}
        self.id_name_dict = {}
        self.id_module_dict = {}
        self.previous = {}
        self.afterward = {}

        self.id_input_shape = {}

        self.module_groups = None
        self.id_groups_index_dict = None

        self.register_lambda_flag = False

        for name, sub_module in model.named_modules():
            self.id_name_dict[id(sub_module)] = name
            self.name_pointer[name] = sub_module
            if isinstance(sub_module, torch.nn.Conv2d) \
                    or isinstance(sub_module, torch.nn.Linear) \
                    or isinstance(sub_module, torch.nn.BatchNorm2d):
                self.hooks.append(sub_module.register_forward_pre_hook(self._pre_forward_relation_hook))
                self.hooks.append(sub_module.register_forward_hook(self._forward_relation_hook))

            else:
                i = 0
                for _ in sub_module.children():
                    i += 1
                    break
                if i == 0:
                    self.hooks.append(sub_module.register_forward_hook(self._trivial_forward_relation_hook))

    def collect_info(self, size_of_inputs):

        dummy = torch.randn((1,)+tuple(size_of_inputs))
        self.module(dummy)
        self._remove()

        for module in self.previous:
            # print(self.id_name_dict[module], [self.id_name_dict[id(x)] for x in self.previous[module]])
            if len(self.previous[module]) == 0:
                self.head.append(self.id_module_dict[module])

        for module in self.afterward:
            if module not in self.previous:
                if module not in self.head:
                    self.head.append(self.id_module_dict[module])

        print('found head:', self.head)

        for module in self.afterward:
            # print(self.id_name_dict[module], [self.id_name_dict[id(x)] for x in self.previous[module]])
            if len(self.afterward[module]) == 0:
                self.tail.append(self.id_module_dict[module])

        for module in self.previous:
            if module not in self.afterward:
                if module not in self.tail:
                    self.tail.append(self.id_module_dict[module])

        print('found tail:', self.tail)

        # for module in self.afterward:
        #     print(self.id_name_dict[module], [self.id_name_dict[id(x)] for x in self.afterward[module]])

    def _remove(self):

        for h in self.hooks:
            h.remove()

    def get_corresponding_bn(self, conv_module):

        if isinstance(conv_module, torch.nn.Conv2d):

            if id(conv_module) in self.afterward:
                next_modules = self.afterward[id(conv_module)]
            else:
                print(conv_module, 'dont have afterward modules.')
                return None

            if len(next_modules) == 1 and isinstance(next_modules[0], torch.nn.BatchNorm2d):
                # print('for', conv_module, 'found', next_modules[0])
                return next_modules[0]
            else:
                return None

        else:
            return None

    def get_next_channel_change_modules(self, module):

        return_list = []
        explore = self.afterward[id(module)]

        while len(explore) > 0:
            a_module = explore[0]
            explore = explore[1:]
            if (isinstance(a_module, torch.nn.Conv2d) and a_module.groups == 1) \
                    or isinstance(a_module, torch.nn.Linear):
                return_list.append(a_module)
            else:
                explore += self.afterward[id(a_module)]

        return return_list

    def get_previous_channel_change_modules(self, module):

        return_list = []
        explore = self.previous[id(module)]

        while len(explore) > 0:
            a_module = explore[0]
            explore = explore[1:]
            if (isinstance(a_module, torch.nn.Conv2d) and a_module.groups == 1) \
                    or isinstance(a_module, torch.nn.Linear):
                return_list.append(a_module)
            else:
                explore += self.previous[id(a_module)]

        return return_list

    def get_modules_with_same_input(self):

        if self.module_groups is None:
            # form [[],[],...]
            module_groups = []
            id_groups_index_dict = {}

            for a_module_id in self.afterward:

                a_module = self.id_module_dict[a_module_id]

                next_conv_modules = self.get_next_channel_change_modules(a_module)

                if next_conv_modules is not None and len(next_conv_modules) > 1:
                    for a_next_module in next_conv_modules:

                        # find the group and merge
                        if id(a_next_module) in id_groups_index_dict:
                            group_index = id_groups_index_dict[id(a_next_module)]
                            for mm in next_conv_modules:
                                if mm not in module_groups[group_index]:
                                    module_groups[group_index].append(mm)
                        else:
                            mark_index = len(module_groups)
                            module_groups.append(next_conv_modules)
                            for mm in next_conv_modules:
                                id_groups_index_dict[id(mm)] = mark_index

            self.module_groups = module_groups
            self.id_groups_index_dict = id_groups_index_dict

        return self.module_groups, self.id_groups_index_dict

    def get_channel_straight_line(self, module):
        return_list = []
        explore = self.afterward[id(module)]

        while len(explore) > 0:
            a_module = explore[0]
            explore = explore[1:]
            if (isinstance(a_module, torch.nn.Conv2d) and a_module.groups == a_module.in_channels) \
                    or isinstance(a_module, torch.nn.BatchNorm2d):
                return_list.append(a_module)
                explore += self.afterward[id(a_module)]
        return return_list

    def _pre_forward_relation_hook(self, module, inputs: JointTensor):
        self.afterward[id(module)] = []
        self.id_module_dict[id(module)] = module
        self.id_input_shape[id(module)] = inputs[0].shape
        try:
            self.previous[id(module)] = inputs[0].input_module_list
            for a_module in inputs[0].input_module_list:
                if id(a_module) not in self.afterward:
                    self.afterward[id(a_module)] = []
                    print('ERROR might happens.')
                self.afterward[id(a_module)].append(module)
        except AttributeError:
            print('Fail to get forward information. Input a Tensor in', module)

    @staticmethod
    def _forward_relation_hook(module, inputs, output) -> JointTensor:
        if isinstance(output[0], torch.Tensor):
            out = JointTensor(output)
            out.record_input_module(module)
            return out
        else:
            output.reset_list()
            output.record_input_module(module)
            return output

    @staticmethod
    def _trivial_forward_relation_hook(module, inputs, output: torch.Tensor) -> JointTensor:
        if isinstance(inputs[0], JointTensor):
            out = JointTensor(output)
            out.input_module_list = inputs[0].input_module_list
            return out
        else:
            out = JointTensor(output)
            print('Initialize JointTensor started from:', module)
            return out

    def get_para_flops(self):

        para = 0
        flops = 0

        for name in self.name_pointer:

            module = self.name_pointer[name]
            if isinstance(module, torch.nn.Conv2d):
                input_shape = list(self.id_input_shape[id(module)])
                weight_shape = list(module.weight.shape)
                para += weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3]
                if module.bias is not None:
                    para += list(module.bias.shape)[0]
                flops += input_shape[2] * input_shape[3] * weight_shape[0] * weight_shape[1]\
                    * weight_shape[2] * weight_shape[3] / module.stride[0] / module.stride[1]

            elif isinstance(module, torch.nn.BatchNorm2d):
                input_shape = list(self.id_input_shape[id(module)])
                para += 2*list(module.weight.shape)[0]
                flops += input_shape[2] * input_shape[3] * 2 * list(module.weight.shape)[0]

            elif isinstance(module, torch.nn.Linear):
                weight_shape = list(module.weight.shape)
                para += weight_shape[0] * weight_shape[1]
                if module.bias is not None:
                    para += list(module.bias.shape)[0]
                flops += weight_shape[0] * weight_shape[1]

        return int(para), int(2*flops)

    def remove_one_channel(self, module, in_mask, out_mask, is_input=True):
        in_num = in_mask if isinstance(in_mask, int) else torch.sum(torch.sign(in_mask))
        out_num = out_mask if isinstance(out_mask, int) else torch.sum(torch.sign(out_mask))
        # print(in_num, out_num)

        if isinstance(module, torch.nn.Conv2d):
            input_shape = list(self.id_input_shape[id(module)])
            weight_shape = list(module.weight.shape)
            if is_input:
                delta_para = out_num * weight_shape[2] * weight_shape[3] / module.groups
                delta_flops = input_shape[2] * input_shape[3] * out_num / module.groups \
                    * weight_shape[2] * weight_shape[3] / module.stride[0] / module.stride[1]
            else:
                delta_para = in_num * weight_shape[2] * weight_shape[3] / module.groups
                if module.bias is not None:
                    delta_para += 1
                delta_flops = input_shape[2] * input_shape[3] * in_num / module.groups\
                    * weight_shape[2] * weight_shape[3] / module.stride[0] / module.stride[1]

        elif isinstance(module, torch.nn.BatchNorm2d):
            input_shape = list(self.id_input_shape[id(module)])
            delta_para = 2
            delta_flops = input_shape[2] * input_shape[3] * 2

        elif isinstance(module, torch.nn.Linear):
            if is_input:
                delta_para = out_num
                delta_flops = out_num
            else:
                delta_para = in_num
                if module.bias is not None:
                    delta_para += 1
                delta_flops = in_num
        else:
            delta_para = 0
            delta_flops = 0
        # print(module, int(delta_para), int(2*delta_flops))
        return int(delta_para), int(2*delta_flops)

    # A rough realization of crldr++, should bigger than the theory value.
    def get_cascade_lambda(self):

        if not self.register_lambda_flag:
            for name in self.name_pointer:
                module = self.name_pointer[name]
                if (isinstance(module, torch.nn.Conv2d) and module.groups == 1) \
                        or (isinstance(module, torch.nn.Linear)):
                    module.register_buffer('norm_lambda', torch.zeros(1, dtype=torch.float))
                    module.register_buffer('cascade_lambda', torch.zeros(1, dtype=torch.float))
            self.register_lambda_flag = True

        for name in self.name_pointer:
            module = self.name_pointer[name]
            if (isinstance(module, torch.nn.Conv2d) and module.groups == 1 and module.kernel_size == (1,1)) or \
                    isinstance(module, torch.nn.Linear):
                bn = self.get_corresponding_bn(module)
                if bn is not None:
                    w = (torch.squeeze(module.weight).t() * bn.weight / torch.sqrt(bn.running_var)).t()
                else:
                    w = torch.squeeze(module.weight)
                u, s, v = torch.svd(w, compute_uv=False)
                module.norm_lambda.data = torch.pow(torch.sum(s) / 2 / list(s.shape)[0], 2)
                print('init:', name, module.norm_lambda)

        # list of module to expand
        search = []
        dealt_list = []
        for a_tail in self.tail:
            if (isinstance(a_tail, torch.nn.Conv2d) and a_tail.groups == 1) \
                    or (isinstance(a_tail, torch.nn.Linear)):
                a_tail.cascade_lambda += 1
                dealt_list.append(a_tail)
                for next_m in self.get_previous_channel_change_modules(a_tail):
                    if next_m not in search:
                        search.append(next_m)
            else:
                modules = self.get_previous_channel_change_modules(a_tail)
                for a_m in modules:
                    if a_m not in dealt_list:
                        a_m.cascade_lambda += 1
                        dealt_list.append(a_m)
                        for next_m in self.get_previous_channel_change_modules(a_m):
                            if next_m not in search:
                                search.append(next_m)

        while len(search) != 0:
            tmp_search = []
            for a_module in search:
                after_modules = self.get_next_channel_change_modules(a_module)
                can_expand = True
                for a_after_module in after_modules:
                    if a_after_module not in dealt_list:
                        can_expand = False
                        break
                if can_expand:
                    exp = len(after_modules)
                    for a_after_module in after_modules:
                        a_module.cascade_lambda.data += \
                            a_after_module.cascade_lambda * a_after_module.norm_lambda
                    a_module.cascade_lambda.data = np.power(2, exp-1) * a_module.cascade_lambda
                    print(self.id_name_dict[id(a_module)], a_module.cascade_lambda)
                    dealt_list.append(a_module)
                    for next_m in self.get_previous_channel_change_modules(a_module):
                        if next_m not in tmp_search:
                            tmp_search.append(next_m)
                else:
                    tmp_search.append(a_module)
            search = tmp_search
