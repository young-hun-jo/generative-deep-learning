import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


# 경사하강법
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS(
        params=[input_img])  # 업데이트할 대상이 input_img이니깐! (그래서 다른 optim 정의 시 model.parameters() 넣은 것)
    return optimizer


def run_style_transfer(vgg16,
                       vgg16_mean,
                       vgg16_std,
                       input_img,
                       base_img,
                       style_img,
                       num_steps=300,
                       content_weight=1,
                       style_weight=1000000,
                       tv_weight=5e-1):
    print("Build style transfer model..")
    model, content_losses, style_losses, total_variation_losses = get_style_model_and_losses(vgg16, vgg16_mean,
                                                                                             vgg16_std, base_img,
                                                                                             style_img)

    input_img.requires_grad_(True)  # enable backpropagation
    model.requires_grad_(False)  # disable backpropagation

    optimizer = get_input_optimizer(input_img)

    # 경사 하강법으로 input_img 업데이트
    print("Optimizing using Gradient Descent...")
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)  # input image pixel 값들 0 ~ 1 사이로 clip

            optimizer.zero_grad()
            model(input_img)  # forward : calculate content/style loss

            content_score = 0
            style_score = 0
            total_variation_score = 0

            for cl in content_losses:
                content_score += cl.loss
            for sl in style_losses:
                style_score += sl.loss
            for tl in total_variation_losses:
                total_variation_score += tl.loss

            content_score *= content_weight
            style_score *= style_weight
            total_variation_score *= tv_weight

            loss = content_score + style_score + total_variation_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"run {run}")
                print(
                    f"Total Loss: {loss.item(): .3f} | Content Loss: {content_score.item(): .3f} | Style Loss: {style_score.item(): .3f} | Total Variation Loss: {total_variation_score.item(): .3f}")
                print()
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)  # 경사하강법 최적화 후, 이미지 픽셀 값 0 ~ 1로 재 clip

    return input_img


def get_style_model_and_losses(vgg16,
                               vgg16_mean,
                               vgg16_std,
                               base_img,
                               style_img):
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    # image normalization layer
    normalization = Normalization(vgg16_mean, vgg16_std).to(device)

    content_losses = []
    style_losses = []
    total_variation_losses = []

    model = nn.Sequential(normalization)

    # vgg16의 모든 layers들을 [입력 -> 출력] 순서로 layer 하나씩 돌면서 내 `model`에 추가
    i = 0
    for layer in vgg16.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            layer = nn.ReLU(inplace=False)  # inplace=True에서 손실 값이 잘 동작하지 않는다는데..?
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError("Unrecognized layer: {}".format(layer.__class__.__name__))
        model.add_module(name, layer)

        # 컨텐츠, 스타일 손실 추가. 추가될 시에 vgg16 모델에서 해당 layer까지 빌드된 상태임
        if name in content_layers:
            syn_img = model(base_img).detach()
            content_loss = ContentLoss(syn_img)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

            total_variation_loss = TotalVariationLoss(syn_img)
            model.add_module("total_variation_loss_{}".format(i), total_variation_loss)
            total_variation_losses.append(total_variation_loss)

        if name in style_layers:
            syn_ftr = model(style_img).detach()
            style_loss = StyleLoss(syn_ftr)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # 컨텐츠, 스타일 손실이 존재하는 레이어의 인덱싱 번호를 알아내기
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            end_idx = i
            break
    model = model[:(end_idx + 1)]
    return model, content_losses, style_losses, total_variation_losses


def gram_matrix(feature):
    b, c, h, w = feature.size()

    feature_map = feature.view(b * c, h * w)  # 2차원으로 변경
    gram_matrix = torch.mm(feature_map, feature_map.t())

    return gram_matrix.div(b * c * h * w)


def verbose_shape(x):
    print(x.shape)


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# 1.컨텐츠 손실
class ContentLoss(nn.Module):
    def __init__(self, syn_img):
        super(ContentLoss, self).__init__()
        self.syn_img = syn_img.detach()

    def forward(self, base_img):
        self.loss = F.mse_loss(base_img, self.syn_img)
        return base_img


class StyleLoss(nn.Module):
    def __init__(self, syn_feature):
        super(StyleLoss, self).__init__()
        self.syn_gram = gram_matrix(syn_feature).detach()

    def forward(self, style_feature):
        style_gram = gram_matrix(style_feature)
        self.loss = F.mse_loss(style_gram, self.syn_gram)
        return style_feature


class TotalVariationLoss(nn.Module):
    def __init__(self, syn_img):
        super(TotalVariationLoss, self).__init__()
        self.syn_img = syn_img

    def forward(self, input_img):
        orig_img = self.syn_img[..., :-1, :-1]
        orig_img_right = self.syn_img[..., :-1, 1:]
        orig_img_down = self.syn_img[..., 1:, :-1]

        right_loss = F.mse_loss(orig_img, orig_img_right)
        down_loss = F.mse_loss(orig_img, orig_img_down)
        self.loss = right_loss + down_loss
        return input_img


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


device = 'cpu'
imsize = 128

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()  # Tensor로 변환하면서 자동으로 0 ~ 255 value를 0 ~ 1사이로 normalize
    ]
)

vgg16_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
vgg16_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
style_img = image_loader("./dataset/images/picasso.jpg")
base_img = image_loader("./dataset/images/dancing.jpg")
input_img = base_img.clone()

# train
output = run_style_transfer(vgg16, vgg16_mean, vgg16_std, input_img, base_img, style_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()