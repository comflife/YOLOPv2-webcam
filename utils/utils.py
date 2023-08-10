# ----------------------------------------------------------------------------
# Copyright (C) [2023] Byounggun Park
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

import datetime
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
import re
import glob
import random
import cv2
import numpy as np
import torch
import torchvision
from sklearn.cluster import KMeans, DBSCAN



logger = logging.getLogger(__name__)


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository

def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOPv2 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """
    Plots a single bounding box on the given image.
    
    Parameters:
    - x: A list or tuple containing the coordinates of the bounding box. Expected format is [x1, y1, x2, y2].
    - img: The image (typically a numpy array) on which the bounding box will be drawn.
    - color (optional): The color of the bounding box in BGR format. If not provided, a random color is used.
    - label (optional): A string that represents the label to be shown near the bounding box.
    - line_thickness (optional): Thickness of the lines used to draw the bounding box. Defaults to 3.
    
    Returns:
    None. The function modifies the input image in-place by drawing the bounding box and label.
    """
    
    # Calculate line thickness based on image dimensions if not provided.
    # The formula is an average of image's width and height multiplied by a small factor.
    # The +1 at the end ensures the thickness is at least 1.
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  
    
    # If color is not provided, generate a random BGR color.
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    # Convert bounding box coordinates to integer values and unpack into top-left (c1) and bottom-right (c2) corners.
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    
    # Draw the rectangle (bounding box) on the image.
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    # If a label is provided, place it above the top-left corner of the bounding box.
    if label:
        # Calculate the font thickness. Ensure it's at least 1.
        tf = max(tl - 1, 1)
        
        # Get the width and height of the label text.
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        
        # Adjust the bottom-right corner of the label's background rectangle.
        # It is shifted left by the width of the text and up by its height.
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

        # Draw the label's background rectangle and the text itself.
        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # Negative thickness means filled rectangle.
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class SegmentationMetric(object):
    '''
    SegmentationMetric class for evaluating the performance of image segmentation models.

    Attributes:
    - imgLabel: Ground truth labels with shape [batch_size, height(144), width(256)]
    - confusionMatrix: Matrix that describes the performance of a classification model
                       [[0(TN),1(FP)],
                        [2(FN),3(TP)]]
    '''
    
    def __init__(self, numClass):
        """
        Initialize the SegmentationMetric with the number of classes.

        Args:
        - numClass: Number of classes for the segmentation task.
        """
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def pixelAccuracy(self):
        """
        Compute overall pixel accuracy across all classes.

        Returns:
        - acc: Overall pixel accuracy.
        """
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def lineAccuracy(self):
        """
        Compute accuracy for the second class (index 1).

        Returns:
        - Acc: Accuracy for class with index 1.
        """
        Acc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1) + 1e-12)
        return Acc[1]

    def classPixelAccuracy(self):
        """
        Compute pixel accuracy for each class.

        Returns:
        - classAcc: Array containing accuracy for each class.
        """
        classAcc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + 1e-12)
        return classAcc

    def meanPixelAccuracy(self):
        """
        Compute mean pixel accuracy across all classes.

        Returns:
        - meanAcc: Mean pixel accuracy.
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        """
        Compute Mean Intersection over Union across all classes.

        Returns:
        - mIoU: Mean IoU across all classes.
        """
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        mIoU = np.nanmean(IoU)
        return mIoU

    def IntersectionOverUnion(self):
        """
        Compute Intersection over Union for the second class (index 1).

        Returns:
        - IoU for class with index 1.
        """
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        return IoU[1]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        """
        Generate a confusion matrix for given predictions and ground truth labels.

        Args:
        - imgPredict: Predicted labels.
        - imgLabel: Ground truth labels.

        Returns:
        - confusionMatrix: Computed confusion matrix.
        """
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        Compute Frequency Weighted Intersection over Union.

        Returns:
        - FWIoU: Frequency Weighted IoU.
        """
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        """
        Update the confusion matrix with a new batch of predictions and labels.

        Args:
        - imgPredict: Predicted labels for the batch.
        - imgLabel: Ground truth labels for the batch.
        """
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        """
        Reset the confusion matrix to zeros.
        """
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


class AverageMeter(object):
    """
    Utility class to keep track of a running average and current value of some metrics.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all internal metrics to zero.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the metrics with a new value.

        Args:
        - val: The new value to update.
        - n: The weighting factor, useful for batched updates.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def _make_grid(nx=20, ny=20):
    """
    Creates a grid of coordinates.

    Args:
    - nx: Width of the grid.
    - ny: Height of the grid.

    Returns:
    - Tensor of size (1, 1, ny, nx, 2) representing the coordinate grid.
    """
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def split_for_trace_model(pred=None, anchor_grid=None):
    """
    Post-processes the prediction tensor for object detection.

    Args:
    - pred: List of prediction tensors.
    - anchor_grid: List of anchor grids for each prediction tensor.

    Returns:
    - Processed predictions tensor.
    """
    z = []
    st = [8,16,32]
    for i in range(3):
        bs, _, ny, nx = pred[i].shape
        # Reshape and permute tensor dimensions
        pred[i] = pred[i].view(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        y = pred[i].sigmoid()
        gr = _make_grid(nx, ny).to(pred[i].device)
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + gr) * st[i]  # Adjust xy coordinates
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # Adjust width and height
        z.append(y.view(bs, -1, 85))
    pred = torch.cat(z, 1)
    return pred

def show_seg_result(img, result, palette=None, img_shape=(480,640), is_demo=False):
    """
    Visualizes segmentation results on an image.

    Args:
    - img: Input image.
    - result: Segmentation result.
    - palette: Color palette for different classes.
    - img_shape: Desired shape of the image for visualization.
    - is_demo: If True, uses demo mode for visualization.

    Returns:
    - Image with overlayed segmentation results.
    """
    if img_shape is not None:
        h, w = img_shape

    # Default color palette if none is provided
    if palette is None:
        palette = np.random.randint(0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    assert palette.shape[0] == 3
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2

    # Colorize the segmentation result
    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        color_area[result[0] == 1] = [0, 0, 255] # Drivable area
        color_area[result[1] == 1] = [0, 255, 255] # lane
        color_area[result[2] == 1] = [225,0,0]
        color_seg = color_area

    # Resize to match the original image shape if provided
    if img_shape is not None:
        color_seg = cv2.resize(color_seg, (w, h), interpolation=cv2.INTER_NEAREST)

    # Convert to BGR for OpenCV
    color_seg = color_seg[..., ::-1]

    # Adjust dimensions if required
    if img.shape[:2] != color_seg.shape[:2]:
        color_seg = cv2.resize(color_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Blend original image with segmentation result
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5

    return img

def show_lane_lines(img, left_fit, right_fit, color=(0, 255, 0), thickness=2):
    """
    Draws lane lines on an image.

    Args:
    - img: Input image.
    - left_fit: Coefficients of the polynomial that fits the left lane line.
    - right_fit: Coefficients of the polynomial that fits the right lane line.
    - color: Color of the lane lines.
    - thickness: Thickness of the lane lines.

    Returns:
    - Image with drawn lane lines.
    """
    if left_fit is None or right_fit is None:
        print("Error: Polynomial fit coefficients are None.")
        return img

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

    # Calculate x values for left and right lanes
    if len(left_fit) == 2:  # linear fit
        leftx = left_fit[0]*ploty + left_fit[1]
        rightx = right_fit[0]*ploty + right_fit[1]
    elif len(left_fit) == 3:  # quadratic fit
        leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    else:
        raise ValueError("left_fit and right_fit should be of length 2 or 3")

    # Convert floating point x coordinates to integers
    leftx = leftx.astype(np.int32)
    rightx = rightx.astype(np.int32)

    # Draw left and right lane lines
    for i in range(img.shape[0]):
        cv2.line(img, (leftx[i], i), (leftx[i], i), color, thickness)
        cv2.line(img, (rightx[i], i), (rightx[i], i), color, thickness)

    return img



def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
        vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img0 = cv2.resize(img0, (1280,720), interpolation=cv2.INTER_LINEAR)
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

def letterbox(img, new_shape=(640, 480), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    #print(sem_img.shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
     
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (dw, dh)

# def detect_stop_line(mask, threshold=20):
#     """
#     Detect stop lines in the segmentation mask based on continuous horizontal pixels.

#     Args:
#     - mask: Input segmentation mask.
#     - threshold: Minimum number of continuous pixels to be considered as a stop line.

#     Returns:
#     - stop_line_mask: Mask with detected stop lines set to 1.
#     """
#     stop_line_mask = np.zeros_like(mask)
#     for i in range(mask.shape[1]):  # iterating through columns now
#         continuous_count = 0
#         for j in range(mask.shape[0]):  # iterating through rows
#             if mask[j, i] == 1:
#                 continuous_count += 1
#                 if continuous_count > threshold:
#                     stop_line_mask[j-continuous_count:j, i] = 1
#                     continuous_count = 0
#             else:
#                 continuous_count = 0
#     return stop_line_mask

# def detect_stop_line(mask, horizontal_threshold=20, vertical_threshold=10):
#     """
#     Detect stop lines in the segmentation mask based on continuous horizontal pixels 
#     and remove long vertical lines.

#     Args:
#     - mask: Input segmentation mask.
#     - horizontal_threshold: Minimum number of continuous pixels in horizontal to be considered as a stop line.
#     - vertical_threshold: Maximum number of continuous pixels in vertical to be considered as not a stop line.

#     Returns:
#     - stop_line_mask: Mask with detected stop lines set to 1.
#     """
#     stop_line_mask = np.zeros_like(mask)
    
#     for i in range(mask.shape[1]):  # iterating through columns
#         continuous_horizontal_count = 0
#         continuous_vertical_count = 0
        
#         for j in range(mask.shape[0]):  # iterating through rows
#             if mask[j, i] == 1:
#                 continuous_horizontal_count += 1
#                 continuous_vertical_count += 1
                
#                 if continuous_horizontal_count > horizontal_threshold:
#                     stop_line_mask[j-continuous_horizontal_count:j, i] = 1
#                     continuous_horizontal_count = 0
                
#                 # If the vertical count exceeds the threshold, reset it
#                 if continuous_vertical_count > vertical_threshold:
#                     stop_line_mask[j-continuous_vertical_count:j, i] = 0
#                     continuous_vertical_count = 0
                    
#             else:
#                 continuous_horizontal_count = 0
#                 continuous_vertical_count = 0
                
#     return stop_line_mask

# import numpy as np

def detect_stop_line(mask, horizontal_threshold=25, vertical_threshold=50): #k-city test vertical_threshold = 45
    stop_line_mask = np.zeros_like(mask)

    # Calculate continuous horizontal count
    horizontal_count = np.zeros_like(mask, dtype=np.int32)
    horizontal_count[:, 0] = mask[:, 0]
    for i in range(1, mask.shape[1]):
        horizontal_count[:, i] = mask[:, i] * (horizontal_count[:, i-1] + 1)

    # Calculate continuous vertical count
    vertical_count = np.zeros_like(mask, dtype=np.int32)
    vertical_count[0, :] = mask[0, :]
    for j in range(1, mask.shape[0]):
        vertical_count[j, :] = mask[j, :] * (vertical_count[j-1, :] + 1)

    # Identify stop lines based on horizontal and vertical counts
    stop_line_mask = np.where((horizontal_count > horizontal_threshold) &
                              (vertical_count <= vertical_threshold), 1, 0)

    return stop_line_mask


def apply_grid_mask(mask, grid_size, grid_range):
    """
    Applies a grid mask to limit computations within specified grid cells.

    Args:
    - mask: Input mask.
    - grid_size: Size of the grid cells.
    - grid_range: Range of grid cells for computations. Specified as a tuple: (start_row, end_row, start_col, end_col).

    Returns:
    - Mask limited to the specified grid cells.
    """
    # Copy the input mask to avoid modifying it directly
    masked = mask.copy()

    # Compute spacing between grid lines
    spacing_x = mask.shape[1] // grid_size
    spacing_y = mask.shape[0] // grid_size

    # Compute the pixel range for computations
    start_x = grid_range[2] * spacing_x
    end_x = grid_range[3] * spacing_x
    start_y = grid_range[0] * spacing_y
    end_y = grid_range[1] * spacing_y

    # Remove mask outside the pixel range
    masked[:start_y, :] = 0
    masked[end_y:, :] = 0
    masked[:, :start_x] = 0
    masked[:, end_x:] = 0

    return masked




# def driving_area_mask(seg = None):
#     da_predict = seg[:, :, 12:372,:]
#     da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=2, mode='bilinear')
#     _, da_seg_mask = torch.max(da_seg_mask, 1)
#     da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

#     return da_seg_mask

def driving_area_mask(seg=None, grid_size=6, grid_range=(0, 6, 0, 6)):
    da_predict = seg[:, :, 12:372,:]
    da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=2, mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

    # Apply grid mask to limit computations within specified grid cells
    da_seg_mask = apply_grid_mask(da_seg_mask, grid_size, grid_range)

    return da_seg_mask


# def lane_line_mask(ll=None):
#     ll_predict = ll[:, :, 12:372,:]
#     ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=2, mode='bilinear')
#     ll_seg_mask = torch.round(ll_seg_mask).squeeze(1)
#     ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

#     # Detect stop lines based on continuous horizontal pixels
#     stop_line_detected = detect_stop_line(ll_seg_mask)

#     # Remove detected stop line from ll_seg_mask
#     ll_seg_mask_without_stopline = np.where(stop_line_detected == 1, 0, ll_seg_mask)


#     return ll_seg_mask_without_stopline, stop_line_detected

def lane_line_mask(ll=None, grid_size=6, grid_range=(0, 6, 0, 6)):
    ll_predict = ll[:, :, 12:372,:]
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=2, mode='bilinear')
    ll_seg_mask = torch.round(ll_seg_mask).squeeze(1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

    # Apply grid mask to limit computations within specified grid cells
    ll_seg_mask = apply_grid_mask(ll_seg_mask, grid_size, grid_range)

    return ll_seg_mask


def detect_lane_with_sliding_window(lane_mask, nwindows=9, margin=100, minpix=50):
    histogram = np.sum(lane_mask[lane_mask.shape[0]//2:,:], axis=0)
    
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    window_height = int(lane_mask.shape[0]//nwindows)
    
    nonzero = lane_mask.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        win_y_low = lane_mask.shape[0] - (window+1)*window_height
        win_y_high = lane_mask.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty

def safe_polyfit(y, x, order):
    try:
        return np.polyfit(y, x, order)
    except TypeError:
        print("Error: inputs to polyfit are not of proper types or shape.")
        return None
    except ValueError:
        print("Error: No valid data points to fit the polynomial.")
        return None



def draw_grid(image, grid_size, color=(0, 255, 0), line_width=1):
    """
    Draws a grid on the image.

    Args:
    - image: Input image.
    - grid_size: Size of the grid cells.
    - color: Color of the grid lines.
    - line_width: Width of the grid lines.

    Returns:
    - Image with overlayed grid.
    """
    # Copy the input image to avoid modifying it directly
    img = image.copy()

    # Compute spacing between grid lines
    spacing_x = img.shape[1] // grid_size
    spacing_y = img.shape[0] // grid_size

    # Draw vertical grid lines
    for x in range(0, img.shape[1], spacing_x):
        cv2.line(img, (x, 0), (x, img.shape[0]), color, line_width)

    # Draw horizontal grid lines
    for y in range(0, img.shape[0], spacing_y):
        cv2.line(img, (0, y), (img.shape[1], y), color, line_width)

    return img

