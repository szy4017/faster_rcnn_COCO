class ROIHead(nn.Module):
    def __init__(self,
                 box_head,
                 roi_pooling,
                 box_predict,
                 box_fg_iou_thresh=0.5,
                 box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512,
                 box_positive_fraction=0.25,
                 box_detections_per_img=100,
                 box_score_thresh=0.05,
                 box_nms_thresh=0.5,
                 iou_type="giou"):
        super(ROIHead, self).__init__()
        self.box_head = box_head
        self.roi_pooling = roi_pooling
        self.box_predict = box_predict
        self.box_fg_iou_thresh = box_fg_iou_thresh
        self.box_bg_iou_thresh = box_bg_iou_thresh
        self.box_batch_size_per_image = box_batch_size_per_image
        self.box_positive_fraction = box_positive_fraction
        self.box_detections_per_img = box_detections_per_img
        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh

        self.proposal_matcher = Matcher(
            self.box_fg_iou_thresh,
            self.box_bg_iou_thresh,
            allow_low_quality_matches=False
        )
        self.box_coder = BoxCoder()
        self.box_loss = IOULoss(iou_type=iou_type)
        self.ce = nn.CrossEntropyLoss()

    def balanced_positive_negative_sampler(self, match_idx):
        sample_size = self.box_batch_size_per_image
        positive_fraction = self.box_positive_fraction
        positive = torch.nonzero(match_idx >= 0, as_tuple=False).squeeze(1)
        negative = torch.nonzero(match_idx == Matcher.BELOW_LOW_THRESHOLD, as_tuple=False).squeeze(1)
        num_pos = int(sample_size * positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = sample_size - num_pos
        num_neg = min(negative.numel(), num_neg)

        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx_per_image = positive[perm1]
        neg_idx_per_image = negative[perm2]
        return pos_idx_per_image, neg_idx_per_image

    def select_training_samples(self, proposals, gt_boxes):
        proposals = [torch.cat([p, g[:, 1:]]) for p, g in zip(proposals, gt_boxes)]
        proposals_ret = list()
        proposal_idx = list()
        for idx, p, g in zip(range(len(proposals)), proposals, gt_boxes):
            if len(g) == 0:
                match_idx = torch.full_like(p[:, 0], fill_value=Matcher.BELOW_LOW_THRESHOLD).long()
            else:
                gt_anchor_iou = box_iou(g[:, 1:], p)
                match_idx = self.proposal_matcher(gt_anchor_iou)
            positive_idx, negative_idx = self.balanced_positive_negative_sampler(match_idx)
            proposal_idx.append((positive_idx, negative_idx, match_idx[positive_idx].long()))
            proposals_ret.append(p[torch.cat([positive_idx, negative_idx])])
        return proposals_ret, proposal_idx

    def compute_loss(self, cls_predicts, box_predicts, proposal_idx, gt_boxes):
        assert proposal_idx is not None and gt_boxes is not None
        all_cls_idx = list()
        positive_mask = list()
        target_boxes = list()
        for prop_idx, gt_box in zip(proposal_idx, gt_boxes):
            p, n, g = prop_idx
            p_cls = gt_box[g][:, 0]
            n_cls = torch.full((len(n),), -1., device=p_cls.device, dtype=p_cls.dtype)
            all_cls_idx.append(p_cls)
            all_cls_idx.append(n_cls)
            mask = torch.zeros((len(p) + len(n),), device=p_cls.device).bool()
            mask[:len(p)] = True
            positive_mask.append(mask)
            target_boxes.append(gt_box[g][:, 1:])

        all_cls_idx = (torch.cat(all_cls_idx) + 1).long()
        positive_mask = torch.cat(positive_mask)
        target_boxes = torch.cat(target_boxes)
        box_loss = self.box_loss(box_predicts[positive_mask], target_boxes).sum() / len(target_boxes)
        cls_loss = self.ce(cls_predicts, all_cls_idx)
        return cls_loss, box_loss

    def post_process(self, cls_predicts, box_predicts, valid_size):
        predicts = list()
        for cls, box, wh in zip(cls_predicts, box_predicts, valid_size):
            box[..., [0, 2]] = box[..., [0, 2]].clamp(min=0, max=wh[0])
            box[..., [1, 3]] = box[..., [1, 3]].clamp(min=0, max=wh[1])
            scores = cls.softmax(dim=-1)
            scores = scores[:, 1:]
            labels = torch.arange(scores.shape[-1], device=cls.device)
            labels = labels.view(1, -1).expand_as(scores)
            boxes = box.unsqueeze(1).repeat(1, scores.shape[-1], 1).reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            inds = torch.nonzero(scores > self.box_score_thresh, as_tuple=False).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            keep = ((boxes[..., 2] - boxes[..., 0]) > 1e-2) & ((boxes[..., 3] - boxes[..., 1]) > 1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = batched_nms(boxes, scores, labels, self.box_nms_thresh)
            keep = keep[:self.box_detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            pred = torch.cat([boxes, scores[:, None], labels[:, None]], dim=-1)
            predicts.append(pred)
        return predicts

    def forward(self, feature, proposals, valid_size, targets=None):
        feature_dict = dict()
        proposal_idx = None
        gt_boxes = None
        for i in range(len(feature) - 1):
            feature_dict['{:d}'.format(i)] = feature[i]
        if self.training:
            assert targets is not None
            gt_boxes = targets['target'].split(targets['batch_len'])
            proposals, proposal_idx = self.select_training_samples(proposals, gt_boxes)
        hw_size = [(s[1], s[0]) for s in valid_size]
        box_features = self.roi_pooling(feature_dict, proposals, hw_size)
        box_features = self.box_head(box_features)
        cls_predict, box_predicts = self.box_predict(box_features)
        box_predicts = self.box_coder.decoder(box_predicts, torch.cat(proposals))
        losses = dict()
        predicts = None
        if self.training:
            assert proposal_idx is not None and gt_boxes is not None
            cls_loss, box_loss = self.compute_loss(cls_predict, box_predicts, proposal_idx, gt_boxes)
            losses['roi_cls_loss'] = cls_loss
            losses['roi_box_loss'] = box_loss
        else:
            if cls_predict.dtype == torch.float16:
                cls_predict = cls_predict.float()
            if box_predicts.dtype == torch.float16:
                box_predicts = box_predicts.float()

            batch_nums = [len(p) for p in proposals]
            cls_predict = cls_predict.split(batch_nums, dim=0)
            box_predicts = box_predicts.split(batch_nums, dim=0)
            predicts = self.post_process(cls_predict, box_predicts, valid_size)
        return predicts, losses, proposals, proposal_idx, box_features