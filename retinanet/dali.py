import os
from contextlib import redirect_stdout
from math import ceil
import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import random
from nvidia.dali import pipeline, ops, types
from nvidia.dali.plugin.pytorch import feed_ndarray
from pycocotools.coco import COCO

class COCOPipeline(pipeline.Pipeline):
    'Dali pipeline for COCO'

    def __init__(self, batch_size, num_threads, path, coco, ids, categories_inv, training, annotations, world, device_id, mean, std, resize, max_size):
        super().__init__(batch_size=batch_size, num_threads=num_threads, device_id = device_id, prefetch_queue_depth=num_threads, seed=42)

        self.path = path
        self.training = training
        self.coco = coco
        self.iter = 0
        self.ids = ids
        self.categories_inv = categories_inv

        self.input = ops.ExternalSource()
        self.input_ids = ops.ExternalSource()
        self.boxes = ops.ExternalSource()
        self.labels = ops.ExternalSource()

        self.reader = ops.COCOReader(annotations_file=annotations, file_root=path, num_shards=world,shard_id=torch.cuda.current_device(), ltrb=True, ratio=True, shuffle_after_epoch=True, save_img_ids=True)
        self.decode_train = ops.nvJPEGDecoderSlice(device="mixed", output_type=types.RGB)
        self.decode_infer = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.bbox_crop = ops.RandomBBoxCrop(device='cpu', ltrb=True, scaling=[0.3, 1.0], thresholds=[0.1,0.3,0.5,0.7,0.9])

        self.bbox_flip = ops.BbFlip(device='cpu', ltrb=True)
        self.img_flip = ops.Flip(device='gpu')
        self.coin_flip = ops.CoinFlip(probability=0.5)

        self.rand1 = ops.Uniform(range=[0.5, 1.5])
        self.rand2 = ops.Uniform(range=[0.875, 1.125])
        self.rand3 = ops.Uniform(range=[-0.5, 0.5])
        if isinstance(resize, list): resize = max(resize)
        self.rand4 = ops.Uniform(range=[800, float(max_size)])
        self.twist = ops.ColorTwist(device='gpu')

        self.resize_train = ops.Resize(device='gpu', interp_type=types.DALIInterpType.INTERP_CUBIC, save_attrs=True)
        self.resize_infer = ops.Resize(device='gpu', interp_type=types.DALIInterpType.INTERP_CUBIC, resize_longer=max_size, save_attrs=True)
        self.pad = ops.Paste(device='gpu', fill_value = 0, ratio=1.1, min_canvas_size=max_size, paste_x=0, paste_y=0)
        self.normalize = ops.CropMirrorNormalize(device='gpu', mean=mean, std=std, crop=max_size, crop_pos_x=0, crop_pos_y=0)

    def define_graph(self):

        #images, bboxes, labels, img_ids = self.reader()
        self.images = self.input()
        self.image_ids = self.input_ids()
        self.bboxes = self.boxes()
        self.labels = self.labels()

        #images = self.images
        #img_ids = self.image_ids
        #bboxes = self.bboxes
        #labels = self.labels
        

        if self.training:
            #crop_begin, crop_size, bboxes, labels = self.bbox_crop(self.bboxes, self.labels)
            #images = self.decode_train(self.images, crop_begin, crop_size)
            images=self.decode_infer(self.images)
            #resize = self.rand4()
            #images, attrs = self.resize_train(images, resize_longer=resize)
            images, attrs = self.resize_infer(images)

            saturation = self.rand1()
            contrast = self.rand1()
            brightness = self.rand2()
            hue = self.rand3()
            #images = self.twist(images, saturation=saturation, contrast=contrast, brightness=brightness, hue=hue)

            #flip = self.coin_flip()
            #bboxes = self.bbox_flip(self.bboxes, horizontal=flip)
            #images = self.img_flip(images, horizontal=flip)


        else:
            images = self.decode_infer(self.images)
            images, attrs = self.resize_infer(images)

        resized_images = images
        images = self.normalize(self.pad(images))

        return images, self.bboxes, self.labels, self.image_ids, attrs, resized_images

    def iter_setup(self):
        # Get next COCO images for the batch
        images, bboxes, labels, ids = [], [], [], []
        overflow = False
        for _ in range(self.batch_size):
            id = int(self.ids[self.iter])
            file_name = self.coco.loadImgs(id)[0]['file_name']
            image = open(self.path + file_name, 'rb')
            bbox, label = self._get_target(id)
 
            images.append(np.frombuffer(image.read(), dtype=np.uint8))
            ids.append(np.array([-1 if overflow else id], dtype=np.float))
            bboxes.append(bbox)
            labels.append(label)           


            overflow = self.iter + 1 >= len(self.ids)
            if not overflow:
                self.iter = (self.iter + 1) % len(self.ids)
        self.feed_input(self.images, images)
        self.feed_input(self.image_ids, ids)
        self.feed_input(self.bboxes, bboxes)
        self.feed_input(self.labels, labels)

    def _get_target(self, id):
        'Get annotations for sample'
        #print("getting target for {}".format(id))
        ann_ids = self.coco.getAnnIds(imgIds=id)
        annotations = self.coco.loadAnns(ann_ids)
        dims = self.coco.loadImgs(id)[0]
        w, h = float(dims['width']), float(dims['height'])
        boxes, categories = [], []
        for ann in annotations:
            if ann['bbox'][2] < 1 and ann['bbox'][3] < 1:
                continue
            boxes.append(ann['bbox'])
            cat = ann['category_id']
            if 'categories' in self.coco.dataset:
                cat = self.categories_inv[cat]
            categories.append(cat)

        if boxes:
            np_bboxes = np.array(boxes, dtype=np.double)
            np_bboxes[:, 0] /= w
            np_bboxes[:, 2] /= w
            np_bboxes[:, 1] /= h
            np_bboxes[:, 3] /= h

            np_bboxes[:, 2] += np_bboxes[:,0]
            np_bboxes[:, 3] += np_bboxes[:,1]
            np_labels = np.array(categories, dtype=np.int32)
            #print([x.shape for x in [np_bboxes, np_labels]])
        else:
            np_bboxes = np.empty(shape=[1,4], dtype=np.double) 
            np_labels = -1* np.ones(shape=[1,], dtype=np.int32)
        return np_bboxes, np_labels


class DaliDataIterator():
    'Data loader for data parallel using Dali'

    def __init__(self, path, resize, max_size, batch_size, stride, world, annotations, training=False):
        self.training = training
        self.resize = resize
        self.max_size = max_size
        self.stride = stride
        self.batch_size = batch_size // world
        self.mean = [255.*x for x in [0.485, 0.456, 0.406]]
        self.std = [255.*x for x in [0.229, 0.224, 0.225]]
        self.world = world
        self.path = path

        # Setup COCO
        with redirect_stdout(None):
            self.coco = COCO(annotations)
        self.ids = sorted(list(self.coco.imgs.keys()))
        self.local_ids = np.array_split(np.array(self.ids), world)[torch.cuda.current_device()]

        if 'categories' in self.coco.dataset:
            self.categories_inv = { k: i for i, k in enumerate(self.coco.getCatIds()) }

        self.pipe = COCOPipeline(batch_size=self.batch_size, num_threads=2, 
            path=path, coco=self.coco, ids=self.local_ids, categories_inv = self.categories_inv, training=training, annotations=annotations, world=world, device_id = torch.cuda.current_device(), mean=self.mean, std=self.std, resize=resize, max_size=max_size)

        self.pipe.build()

    def __repr__(self):
        return '\n'.join([
            '    loader: dali',
            '    resize: {}, max: {}'.format(self.resize, self.max_size),
        ])

    def __len__(self):
        return ceil(len(self.local_ids) / self.batch_size)

    def __iter__(self):
        for _ in range(self.__len__()):
            #print("about to run pipe")
            data, ratios, ids, num_detections = [], [], [], []
            dali_data, dali_boxes, dali_labels, dali_ids, dali_attrs, dali_resize_img = self.pipe.run()
           # print("got pipe outputs")
            for l in range(len(dali_boxes)):
                num_detections.append(dali_boxes.at(l).shape[0])

            pyt_targets = -1 * torch.ones([len(dali_boxes), max(max(num_detections),1), 5])

            for batch in range(self.batch_size):
                id = int(dali_ids.at(batch)[0])
                if id < 0: break
                
                # Convert dali tensor to pytorch
                dali_tensor = dali_data.at(batch)
                tensor_shape = dali_tensor.shape()

                datum = torch.zeros(dali_tensor.shape(), dtype=torch.float, device=torch.device('cuda'))
                c_type_pointer = ctypes.c_void_p(datum.data_ptr())
                dali_tensor.copy_to_external(c_type_pointer)

                # Calculate image resize ratio to rescale boxes
                prior_size = dali_attrs.as_cpu().at(batch)
                resized_size = dali_resize_img.at(batch).shape()
                ratio = max(resized_size) / max(prior_size)

                if self.training:
                    # Rescale boxes
                    b_arr = dali_boxes.at(batch)
                    num_dets = b_arr.shape[0]
                    if num_dets is not 0:
                        pyt_bbox = torch.from_numpy(b_arr).float()

                        pyt_bbox[:,0] *= float(prior_size[1])
                        pyt_bbox[:,1] *= float(prior_size[0])
                        pyt_bbox[:,2] *= float(prior_size[1])
                        pyt_bbox[:,3] *= float(prior_size[0])
                        # (l,t,r,b) ->  (x,y,w,h) == (l,r, r-l, b-t)
                        pyt_bbox[:,2] -= pyt_bbox[:,0]
                        pyt_bbox[:,3] -= pyt_bbox[:,1]
                        pyt_targets[batch,:num_dets,:4] = pyt_bbox * ratio

                    # Arrange labels in target tensor
                    l_arr = dali_labels.at(batch)
                    if num_dets is not 0:
                        pyt_label = torch.from_numpy(l_arr).float()
                        pyt_label -= 1 #Rescale labels to [0,79] instead of [1,80]
                        pyt_targets[batch,:num_dets, 4] = pyt_label.squeeze()

                ids.append(id)
                data.append(datum.unsqueeze(0))
                ratios.append(ratio)

            data = torch.cat(data, dim=0)

            if self.training:
                pyt_targets = pyt_targets.cuda(non_blocking=True)

                yield data, pyt_targets

            else:
                ids = torch.Tensor(ids).int().cuda(non_blocking=True)
                ratios = torch.Tensor(ratios).cuda(non_blocking=True)

                yield data, ids, ratios
