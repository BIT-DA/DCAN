# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch.nn as nn


class GradCAM(object):
    """
    1: The network does not need to update the gradients
    2: Use the score of the target category for back propagation
    """

    def __init__(self, net, layer_name, class_name_list):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
        self.class_name_list = class_name_list

    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        """
        :param inputs: [1,3,H,W]
        :param index: class_id
        """
        self.net.zero_grad()
        output, _ = self.net(inputs)
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        softmax_layer = nn.Softmax(dim=1)
        softmax_output = softmax_layer(output)
        score = softmax_output[0][index]
        print("class_id= {}  class_name= {}  softmax_score= {:.5f}".format(index, self.class_name_list[index], score))

        # print top-k accuracy
        pred_k, indices_k  = softmax_output.topk(5, 1, True, True)
        for i in range(5):
            print("{} {:.6f}".format(self.class_name_list[indices_k[0][i]], pred_k[0][i].cpu().data.numpy()))

        target.backward()
        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].cpu().data.numpy()

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # normalize
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (224, 224))
        return cam, index, self.class_name_list[index], score
