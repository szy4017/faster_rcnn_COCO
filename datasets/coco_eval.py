# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch
from matplotlib import pyplot as plt
import cv2 as cv

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from utils.misc import all_gather


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, score_threshold=0.5):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        self.coco_eval_state = {}   ## 存储eval state的相关信息
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval_state[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.eval_imgs_state = {h: [] for h in iou_types}   ## 存储eval state的相关信息
        self.score_threshold = score_threshold  ## 可视化预测结果的置信度阈值
        self.save_file = True

    ## update将coco_dt存入coco_eval中，然后进行coco_eval.accumulate()和coco_eval.summarize()就可以得到评估结果了
    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)   ## 得到COCO形式的results<dict>
            self.show_result(results)   # show the image results of class prediction and state prediction

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]
            coco_eval_state = self.coco_eval_state[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval_state.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            coco_eval_state.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)
            img_ids_state, eval_imgs_state = evaluatestate(coco_eval_state)   # 评估入侵状态标签

            self.eval_imgs[iou_type].append(eval_imgs)
            self.eval_imgs_state[iou_type].append(eval_imgs_state)
            pass

    # 显示预测结果
    def show_result(self, results):
        """
        可视化预测结果，包括class prediction和state prediction

        :params results: 预测结果
        """
        res_list = list()
        for res in results:
            if res['score'] > self.score_threshold:
                res_list.append(res)
        results = res_list

        # load prediction annotations
        coco = self.coco_gt
        cocoRes = coco.loadRes(results)
        catIds = cocoRes.getCatIds(catNms='pedestrian')
        imgIds = cocoRes.getImgIds(catIds=catIds)
        img_info = cocoRes.loadImgs(imgIds[np.random.randint(0, len(imgIds))])
        annIds = cocoRes.getAnnIds(imgIds=img_info[0]['id'])
        anns = cocoRes.loadAnns(annIds)
        # load ground truth annotation
        annIds_gt = coco.getAnnIds(imgIds=img_info[0]['id'])
        anns_gt = coco.loadAnns(annIds_gt)

        # load image
        root_path = '/data/szy4017/data/intruscapes/images/val'
        file_name = img_info[0]['file_name']
        imgPath = os.path.join(root_path, file_name)
        print(imgPath)
        img = cv.imread(imgPath)
        img_rgb = bgr2rgb(img)

        # plt
        plt.title('Class Prediction')
        plt.imshow(img_rgb)
        cocoRes.showBBox(anns)
        plt.savefig('./misc/cls_pred_{}.png'.format(img_info[0]['id']))
        plt.show()
        plt.title('State Prediction')
        plt.imshow(img_rgb)
        cocoRes.showIntrusion(anns)
        plt.savefig('./misc/sta_pred_{}.png'.format(img_info[0]['id']))
        plt.show()
        plt.title('Class Ground Truth')
        plt.imshow(img_rgb)
        coco.showBBox(anns_gt)
        plt.savefig('./misc/cls_gt_{}.png'.format(img_info[0]['id']))
        plt.show()
        plt.title('State Ground Truth')
        plt.imshow(img_rgb)
        coco.showIntrusion(anns_gt)
        plt.savefig('./misc/sta_gt_{}.png'.format(img_info[0]['id']))
        plt.show()

        pass



    # 将所有self.eval_imgs的数据进行同步
    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    # 将所有self.eval_imgs_state的数据进行同步
    # 基于synchronize_between_processes修改得到
    def synchronize_between_processes_state(self):
        for iou_type in self.iou_types:
            self.eval_imgs_state[iou_type] = np.concatenate(self.eval_imgs_state[iou_type], 2)
            create_common_coco_eval(self.coco_eval_state[iou_type], self.img_ids, self.eval_imgs_state[iou_type])

    # 将coco_eval的值进行累计
    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    # 将coco_eval_state的值进行累计
    # 基于accumulate修改得到
    def accumulate_state(self):
        for coco_eval in self.coco_eval_state.values():
            coco_eval.accumulate_state()

    # 总结eval结果，并输出最终结果
    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    # 总结eval结果，并输出最终结果
    # 基于summarize修改得到
    def summarize_state(self):
        for iou_type, coco_eval in self.coco_eval_state.items():
            print("State IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    # 将预测模型的预测结果转换成COCO形式
    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            # 增加s_scores和s_labels，表示入侵状态预测的状态类别标签和置信度
            s_scores = prediction["s_scores"].tolist()
            s_labels = prediction["s_labels"].tolist()

            # 生成COCO形式的字典，增加state和state_score字段
            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                        "state": s_labels[k],
                        "state_score": s_scores[k]
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results

# 用于图像的色彩通道转换
def bgr2rgb(img):
    # 用cv自带的分割和合并函数
    B, G, R = cv.split(img)
    return cv.merge([R, G, B])

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


# 设置了自己的evaluate函数，而没有采用COCO API中的COCOeval.evaluate，
# 但是调用了COCOeval.evaluateImg进行单张图片单个类别的evaluate
def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    # 计算IoU，保存在self.ious中，evaluateImg是根据self.ious的数据进行评估匹配的
    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId, mode='cls')
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


# 设置了自己的evaluate函数，而没有采用COCO API中的COCOeval.evaluate，
# 但是调用了COCOeval.evaluateImg进行单张图片单个类别的evaluate
# 基于evaluate()修改的evaluatestate()，对state标签进行evaluate
def evaluatestate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    p.staIds = list(np.unique(p.staIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    staIds = p.staIds

    # 计算IoU，保存在self.ious中，evaluateImg是根据self.ious的数据进行评估匹配的
    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, staId): computeIoU(imgId, staId, mode='sta')
        for imgId in p.imgIds
        for staId in staIds}

    evaluateImgState = self.evaluateImgState
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImgState(imgId, staId, areaRng, maxDet)
        for staId in staIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(staIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
