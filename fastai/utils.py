from data import *
import cv2
import random
import torchtoolbox.transform as transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from data import HAIRS
import os

resolution = 456  # orignal res for B5
input_res = 384


class AdvancedHairAugmentation:

    """
    Impose an image of a hair to the target image

    Args:
        hairs (int): maximum number of hairs to impose
        hairs_folder (str): path to the folder with hairs images
    """

    def __init__(self, hairs: int = 5, hairs_folder: str = ""):
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        n_hairs = random.randint(0, self.hairs)

        if not n_hairs:
            return img

        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if "png" in im]

        for _ in range(n_hairs):
            hair = cv2.imread(
                os.path.join(self.hairs_folder, random.choice(hair_images))
            )
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width]

            # Creating a mask and inverse mask
            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of hair in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of hair from hair image.
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            # Put hair in ROI and modify the target image
            dst = cv2.add(img_bg, hair_fg)

            img[roi_ho : roi_ho + h_height, roi_wo : roi_wo + h_width] = dst

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, hairs_folder="{self.hairs_folder}")'


class DrawHair:

    """
    Draw a random number of pseudo hairs

    Args:
        hairs (int): maximum number of hairs to draw
        width (tuple): possible width of the hair in pixels
    """

    def __init__(self, hairs: int = 4, width: tuple = (1, 2)):
        self.hairs = hairs
        self.width = width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        if not self.hairs:
            return img

        width, height, _ = img.shape

        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0, width), random.randint(0, height // 2))
            # The end of the line
            end = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)  # color of the hair. Black.
            cv2.line(
                img, origin, end, color, random.randint(self.width[0], self.width[1])
            )

        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(hairs={self.hairs}, width={self.width})"


class Microscope:

    """
    Cutting out the edges around the center circle of the image
    Imitating a picture, taken through the microscope

    Args:
        p (float): probability of applying an augmentation
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to apply transformation to.

        Returns:
            PIL Image: Image with transformation.
        """
        if random.random() < self.p:
            circle = cv2.circle(
                (np.ones(img.shape) * 255).astype(np.uint8),  # image placeholder
                (img.shape[0] // 2, img.shape[1] // 2),  # center point of circle
                random.randint(img.shape[0] // 2 - 3, img.shape[0] // 2 + 15),  # radius
                (0, 0, 0),  # color
                -1,
            )

            mask = circle - 255
            img = np.multiply(img, mask)

        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


def get_transforms():
    train_transform = transforms.Compose(
        [
            AdvancedHairAugmentation(hairs_folder=f"{HAIRS}"),
            transforms.RandomResizedCrop(size=SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            Microscope(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, test_transform


def get_train_transforms():
    return A.Compose(
        [
            A.JpegCompression(p=0.5),
            A.Resize(384, 384),
            A.Rotate(limit=80, p=1.0),
            A.OneOf(
                [A.OpticalDistortion(), A.GridDistortion(), A.IAAPiecewiseAffine()]
            ),
            A.RandomSizedCrop(
                min_max_height=(int(resolution * 0.5), input_res),
                height=resolution,
                width=resolution,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(p=0.3),
            A.OneOf([A.RandomBrightnessContrast(), A.HueSaturationValue(),]),
            A.Cutout(
                num_holes=8,
                max_h_size=resolution // 8,
                max_w_size=resolution // 8,
                fill_value=0,
                p=0.3,
            ),
            A.Normalize(),
            ToTensorV2(),
        ],
        p=1.0,
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.CenterCrop(height=resolution, width=resolution, p=1.0),
            A.Normalize(),
            ToTensorV2(),
        ],
        p=1.0,
    )


def get_tta_transforms():
    return A.Compose(
        [
            A.JpegCompression(p=0.5),
            A.RandomSizedCrop(
                min_max_height=(int(resolution * 0.9), int(resolution * 1.1)),
                height=resolution,
                width=resolution,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ],
        p=1.0,
    )

