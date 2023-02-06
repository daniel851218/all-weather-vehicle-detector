import torch
import numpy as np

from imgaug import augmenters as iaa

class ImgAugTransform():
    def __init__(self):
        self.aug_noise_gray = iaa.OneOf([
            iaa.Identity(),
            iaa.AdditiveGaussianNoise(loc=0, scale=(2.5, 7.5)),
            iaa.SaltAndPepper(p=0.01),
            iaa.CoarseDropout(p=(0.001, 0.005), size_percent=(0.1, 0.2)),
            iaa.CoarseSaltAndPepper(p=(0.001, 0.005), size_percent=(0.1, 0.2))
        ])

        self.aug_noise_color = iaa.OneOf([
            iaa.Identity(),
            iaa.AdditiveGaussianNoise(loc=0, scale=(2.5, 7.5), per_channel=True),
            iaa.SaltAndPepper(p=0.01, per_channel=True),
            iaa.CoarseDropout(p=(0.001, 0.005), size_percent=(0.1, 0.2), per_channel=True),
            iaa.CoarseSaltAndPepper(p=(0.001, 0.005), size_percent=(0.1, 0.2), per_channel=True)
        ])

        self.aug_blur = iaa.OneOf([
            iaa.Identity(),
            iaa.GaussianBlur(sigma=(0.1, 1.0)),
            iaa.MotionBlur(k=3, angle=(-45, 45))
        ])

        self.aug_intensity = iaa.OneOf([
            iaa.Identity(),
            iaa.MultiplyAndAddToBrightness(),
            iaa.Grayscale(alpha=(0.0, 0.5)),
            iaa.GammaContrast(),
            iaa.GammaContrast(per_channel=True),
        ])

        self.aug = iaa.Sequential([
            iaa.OneOf([self.aug_noise_gray, self.aug_noise_color, iaa.Identity()]),
            iaa.OneOf([self.aug_blur, self.aug_intensity, iaa.Identity()])
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        img = torch.tensor(img).permute([2, 0, 1])
        return img / 255.0