import cv2
import random
import numpy as np
import torch

from torchvision.transforms import functional as F


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            img = sample['img']
            _, w_img, _ = img.shape
            sample['img'] = cv2.flip(img, 1)
            sample['annot'][:, [0, 2]] = w_img - sample['annot'][:, [2, 0]]
        return sample

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            img = sample['img']
            w_img, _, _ = img.shape
            sample['img'] = cv2.flip(img, 0)
            sample['annot'][:, [1, 3]] = w_img - sample['annot'][:, [3, 1]]
        return sample


# TODO
# class Rotation(object):
#     def __init__(self, p=0.25):
#         self.p = p
#
#     def __call__(self, sample):
#         if random.random() < self.p:
#             sample['img'] = cv2.rotate(sample['img'], cv2.ROTATE_90_CLOCKWISE)
#             sample['annots']['boxes'][:, [1, 3]], sample['annots']['boxes'][:, [0, 2]] = \
#                 sample['annots']['boxes'][:, [2, 0]], sample['annots']['boxes'][:, [1, 3]]
#         elif random.random() < self.p * 2:
#             sample['img'] = cv2.rotate(sample['img'], cv2.ROTATE_180)
#             sample['annots']['boxes'][:, [0, 2]] = sample['annots']['boxes'][:, [2, 0]]
#             sample['annots']['boxes'][:, [1, 3]] = sample['annots']['boxes'][:, [3, 1]]
#         elif random.random() < self.p * 3:
#             sample['img'] = cv2.rotate(sample['img'], cv2.ROTATE_90_COUNTERCLOCKWISE)
#             sample['annots']['boxes'][:, [1, 3]], sample['annots']['boxes'][:, [0, 2]] = \
#                 sample['annots']['boxes'][:, [2, 0]], sample['annots']['boxes'][:, [1, 3]]
#         return sample


class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if self.p > np.random.random():
            img = sample['img']
            y_shape, x_shape, _ = img.shape
            x = np.random.randint(low=0, high=x_shape // 4)
            y = np.random.randint(low=0, high=y_shape // 4)

            x_len = np.random.randint(low=x_shape // 2, high=x_shape - x)
            y_len = np.random.randint(low=y_shape // 2, high=y_shape - y)

            img = img[y: y + y_len, x: x + x_len]
            dellist = []
            for i, box in enumerate(sample['annot']):
                if box[i][0] >= x + x_len or box[i][2] <= x or box[i][1] >= y + y_len or box[i][3] <= y:
                    dellist.append(i)
            if len(dellist) == sample['annot'].shape[0]:
                return sample
            sample['annot'] = np.delete(sample['annot'], dellist, 0)

            sample['annot'][:, [0, 2]] = np.clip(sample['annot'][:, [0, 2]] - x, a_min=0, a_max= x_len)
            sample['annot'][:, [1, 3]] = np.clip(sample['annot'][:, [1, 3]] - y, a_min=0, a_max= y_len)
            sample['img'] = img
        return sample


class RandomAffine(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w_img, h_img))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
        return img, bboxes


class CheckValid(object):
    def __call__(self, sample):
        boxes = sample['annot']
        dellist = np.where(np.logical_or(boxes[:, 0] >= boxes[:, 2], boxes[:, 1] >= boxes[:, 3]))
        sample['annot'] = np.delete(sample['annot'], dellist, axis=0)
        return sample


class Resizer(object):
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': new_image, 'annot': annots, 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}
        return sample


class Normalizer(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}

class Mixup(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):
        if random.random() > self.p:
            lam = np.random.beta(1.5, 1.5)
            img = lam * img_org + (1 - lam) * img_mix
            bboxes_org = np.concatenate(
                [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=1
            )
            bboxes_mix = np.concatenate(
                [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=1
            )
            bboxes = np.concatenate([bboxes_org, bboxes_mix])

        else:
            img = img_org
            bboxes = np.concatenate(
                [bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1
            )

        return img, bboxes


class LabelSmooth(object):
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes


class ToTensor(object):
    def __call__(self, sample):
        numpy_img = sample['img']
        numpy_img = numpy_img.transpose((2, 0, 1))
        numpy_img = np.asarray(numpy_img, dtype=np.float32) / 255
        sample['img'] = torch.from_numpy(numpy_img).float()
        sample['annot'] = torch.from_numpy(sample['annot']).float()
        return sample