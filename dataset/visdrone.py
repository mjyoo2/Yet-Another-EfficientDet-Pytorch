import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import torch
import numpy as np

from torchvision import transforms


class VisDroneDataset(Dataset):
    def __init__(self, root_dir, set, transform=None):
        path = os.path.join(root_dir, set)
        self.db = VisDroneDataBase(path)
        self.transform = transform

    def __getitem__(self, index):
        numpy_img, annotations = self.db[index]
        [bbox, cls_label, _] = annotations
        data = dict()
        data['img'] = numpy_img
        if not bbox:
            bbox = [[-1, -1, -1, -1]]
        if not cls_label:
            cls_label = [[-1]]
        bbox = np.asarray(bbox, dtype=np.float32)
        cls_label = np.asarray(cls_label, dtype=np.long).reshape(bbox.shape[0], 1)
        data['annot'] = np.concatenate([bbox, cls_label], axis=1)
        if self.transform:
            return self.transform(data)
        return data

    def __len__(self):
        return len(self.db)


class VisDroneDataBase(object): # 경로 주면 만들어짐?
    def __init__(self, path, ignore_0=True):
        self.path = path  # 데이터 경로?
        self.image_path = self.path + "/images/"
        self.annotation_path = self.path + "/annotations/"
        self.prefix = os.listdir(self.image_path)  # 경로 내의 모든 파일명 리스트로
        for i, s in enumerate(self.prefix):
            self.prefix[i] = s.replace(".jpg", ".{}")
        self.ignore_0 = ignore_0

    def __len__(self):
        return len(self.prefix)

    def __getitem__(self, index):  # 클래스를 인덱스 붙여서 사용할수 있게 하는거
        prefix = self.prefix[index]  #
        image_path = self.image_path + prefix.format("jpg")  # fromat은 위에 보면 prefix에 {} 넣은거 있는데 이거 jpg로 다시 채우기
        annotation_path = self.annotation_path + prefix.format("txt")  # 아무튼 경로? 여러줄?
        annotation = VisDroneDataBase.read_annotation(annotation_path)  # 이러면 세 개의 리스트 묶은 리스트 생김
        image = cv2.imread(image_path)
        return image, annotation

    @staticmethod
    def read_annotation(path, ignore_0=True):
        with open(path, "r") as f:
            annotation_raw = f.readlines()
        bboxes = []
        labels = []
        infos = []
        for line in annotation_raw:
            # remove new line characters
            line = line.replace("\n", "")
            # parse and append
            try:  # 실행할 때 ValueError 생길 때만 except 넘어가는 구조. 오류 없으면 아랫줄만 실행
                bbox, label, info = VisDroneDataBase.parse_annotation_line(line)
            except ValueError:
                # no label case
                continue
            # negative label case
            if label < 0:  # 무슨 경우?
                continue
            if ignore_0 and label == 0:  # 무시하는 영역(ignored region)
                continue
            else:
                bboxes.append(bbox)
                labels.append(label)
                infos.append(info)
        return [bboxes, labels, infos]

    @staticmethod
    def parse_annotation_line(line):  # 텍스트 리스트로 바꾼 후 정보 추출
        line = line.split(",")
        [bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion] = line
        # 위에 좌표는 적당히 냅두고 뒤에 4개 뽑기 위함
        bbox = line[0:4]  # 저 범위가 좌표 있는 부분. 위에 형태 참고
        bbox = list(map(np.float32, bbox))
        # [x, y, w, h] -> [x1, y1, x2, y2]
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]

        infos = [float(score), float(truncation), float(occlusion)]  # 아무튼 세 가지 정보
        return bbox, int(category), infos  # category 값이 label로
