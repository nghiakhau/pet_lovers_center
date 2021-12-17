import torch
from torch import nn
from torchvision import models
import re


class PetLoverCenter(nn.Module):
    def __init__(self, config, logging=None):
        super(PetLoverCenter, self).__init__()

        # embed img features
        self.embeddings = nn.Embedding(num_embeddings=config.num_features * 2,
                                       embedding_dim=config.embedding_dim)

        # encode img
        pretrained = True if config.pretrained == 1 else False
        print("Encode img using {} model. Pretrained = {}".format(
            config.img_model_name, pretrained))
        if re.match(r"^vgg$", config.img_model_name, re.IGNORECASE):
            self.encode_img = models.vgg16_bn(
                pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
            self.encode_img = nn.Sequential(
                *[*self.encode_img.children()][:-2])
            self.encode_img.add_module(
                "Adaptive Average Pool", nn.AdaptiveAvgPool2d(
                    output_size=(2, 2)))
            self.encode_img.add_module("Flatten", nn.Flatten())
            in_features = self.encode_img[-3][41].num_features * 2 * 2
            print("Number of out features of '{}' model = {}".format(
                config.img_model_name, in_features))

        elif re.match(r"^mobilenet$", config.img_model_name, re.IGNORECASE):
            self.encode_img = models.mobilenet_v2(
                pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
            self.encode_img = nn.Sequential(
                *[*self.encode_img.children()][:-1])
            self.encode_img.add_module(
                "Adaptive Average Pool", nn.AdaptiveAvgPool2d(
                    output_size=(1, 1)))
            self.encode_img.add_module("Flatten", nn.Flatten())
            in_features = self.encode_img[-3][-1][1].num_features
            print("Number of out features of '{}' model = {}".format(
                config.img_model_name, in_features))

        elif re.match(r"^densenet$", config.img_model_name, re.IGNORECASE):
            self.encode_img = models.densenet169(
                pretrained=pretrained, progress=True)
            if pretrained:
                self.freeze()
            self.encode_img = nn.Sequential(
                *[*self.encode_img.children()][:-1])
            self.encode_img.add_module(
                "Adaptive Average Pool", nn.AdaptiveAvgPool2d(
                    output_size=(1, 1)))
            self.encode_img.add_module("Flatten", nn.Flatten())
            in_features = self.encode_img[0].norm5.num_features
            print("Number of out features of '{}' model = {}".format(
                config.img_model_name, in_features))

        else:
            self.encode_img = models.resnet50(
                pretrained=pretrained, progress=True)
            if pretrained:
                print('Freeze pretrained weight')
                if logging:
                    logging.info('Freeze pretrained weight')
                self.freeze()
            self.encode_img = nn.Sequential(
                *[*self.encode_img.children()][:-1])
            self.encode_img.add_module("Flatten", nn.Flatten())
            # 2048
            in_features = self.encode_img[-3][2].bn3.num_features
            print("Number of out features of '{}' model = {}".format(
                config.img_model_name, in_features))

        classifier_dims = [
            in_features + config.num_features * config.embedding_dim]
        classifier_dims.extend(config.classifier_dims)
        print("Encode img features with the following layers: {}".format(
            classifier_dims))

        self.classifier = nn.Sequential()
        for i, (d_in, d_out) in enumerate(
                zip(classifier_dims[:-1], classifier_dims[1:])):
            self.classifier.add_module(
                "DROPOUT_" + str(i + 1), nn.Dropout(config.dropout))
            self.classifier.add_module(
                "FC_" + str(i + 1), nn.Linear(
                    in_features=d_in, out_features=d_out))
            if i != len(classifier_dims) - 2:
                self.classifier.add_module(
                    "BN_" + str(i + 1), nn.BatchNorm1d(
                        num_features=d_out, eps=config.eps))
                self.classifier.add_module("ACT_" + str(i + 1), nn.ReLU())
            else:
                self.classifier.add_module("ACT_" + str(i + 1), nn.Sigmoid())
        print(self.classifier)
        # self.init_weights()

    def init_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)

    def freeze(self):
        for params in self.parameters():
            params.requires_grad = False

    def encode(self, img, x):
        img = self.encode_img(img)
        x = self.embeddings(x)
        # x = torch.sum(x, -2)
        x = x.view(img.size()[0], -1)
        x = self.classifier(torch.cat((img, x), 1))
        # return x[:, 0], x[:, 1]
        return x

    def forward(self, img, x, training):
        # mu, logvar = self.encode(img, x)
        # if training:
        #     std = torch.exp(0.5 * logvar)
        #     eps = torch.randn_like(std)
        #     return eps.mul(std).add_(mu), mu, logvar
        # else:
        #     return mu, mu, logvar
        return self.encode(img, x)
