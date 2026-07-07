import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    """
    Edge Loss based on Sobel filters.
    Encourages the network to reconstruct image edges.
    """

    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        )

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            dtype=torch.float32
        )

        self.register_buffer(
            "sobel_x",
            sobel_x.view(1, 1, 3, 3)
        )

        self.register_buffer(
            "sobel_y",
            sobel_y.view(1, 1, 3, 3)
        )

    def gradient(self, image):


        sobel_x = self.sobel_x.to(image.device)
        sobel_y = self.sobel_y.to(image.device)
        gradients = []

        for c in range(image.shape[1]):

            channel = image[:, c:c+1]

            gx = F.conv2d(
                channel,
                sobel_x,
                padding=1
            )

            gy = F.conv2d(
                channel,
                sobel_y,
                padding=1
            )

            grad = torch.sqrt(
                gx ** 2 + gy ** 2 + 1e-6
            )

            gradients.append(grad)

        return torch.cat(
            gradients,
            dim=1
        )

    def forward(self, prediction, target):

        pred_grad = self.gradient(prediction)
        target_grad = self.gradient(target)

        return F.l1_loss(
            pred_grad,
            target_grad
        )


class CombinedLoss(nn.Module):
    """
    L1 Loss + Edge Loss
    """

    def __init__(
        self,
        edge_weight=0.20
    ):

        super().__init__()

        self.l1 = nn.L1Loss()

        self.edge = EdgeLoss()

        self.edge_weight = edge_weight

    def forward(
        self,
        prediction,
        target
    ):

        l1_loss = self.l1(
            prediction,
            target
        )

        edge_loss = self.edge(
            prediction,
            target
        )

        total_loss = (
            l1_loss
            +
            self.edge_weight * edge_loss
        )

        return total_loss
    



from torchvision.models import vgg16, VGG16_Weights


class PerceptualLoss(nn.Module):

    def __init__(self):

        super().__init__()

        vgg = vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1
        ).features[:16]

        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg.eval()

        self.l1 = nn.L1Loss()

    def forward(self, prediction, target):

        device = prediction.device

        self.vgg = self.vgg.to(device)

        pred_feat = self.vgg(prediction)

        target_feat = self.vgg(target)

        return self.l1(
            pred_feat,
            target_feat
        )