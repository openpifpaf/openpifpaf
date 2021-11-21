import torch

from .heads import HeadNetwork, CompositeField4


class TBaseSingleImage(HeadNetwork):
    """Filter the feature map so that they can be used by single image loss.

    Training: only apply loss to image 0 of an image pair of image 0 and 1.
    Evaluation with forward tracking pose: only keep image 0.
    Evaluation with full tracking pose: keep all but stack group along feature dim.
    """
    forward_tracking_pose = True
    tracking_pose_length = 2

    def __init__(self, meta, in_features):
        super().__init__(meta, in_features)
        self.head = CompositeField4(meta, in_features)

    def forward(self, *args):
        x = args[0]

        if self.training:
            x = x[::2]
        elif self.forward_tracking_pose:
            x = x[::self.tracking_pose_length]

        x = self.head(x)

        if not self.training and not self.forward_tracking_pose:
            # full tracking pose eval
            # TODO: stack batch dimension in feature dimension and adjust
            # meta information (make it a property to dynamically return
            # a different meta for evaluation)
            raise NotImplementedError

        return x


class Tcaf(HeadNetwork):
    """Filter the feature map so that they can be used by single image loss.

    Training: only apply loss to image 0 of an image pair of image 0 and 1.
    Evaluation with forward tracking pose: only keep image 0.
    Evaluation with full tracking pose: keep all.
    """
    tracking_pose_length = 2
    reduced_features = 512

    _global_feature_reduction = None
    _global_feature_compute = None

    def __init__(self, meta, in_features):
        super().__init__(meta, in_features)

        if self._global_feature_reduction is None:
            self.__class__._global_feature_reduction = torch.nn.Sequential(
                torch.nn.Conv2d(in_features, self.reduced_features,
                                kernel_size=1, bias=True),
                torch.nn.ReLU(inplace=True),
            )
        self.feature_reduction = self._global_feature_reduction

        if self._global_feature_compute is None:
            self.__class__._global_feature_compute = torch.nn.Sequential(
                torch.nn.Conv2d(self.reduced_features * 2, self.reduced_features * 2,
                                kernel_size=1, bias=True),
                torch.nn.ReLU(inplace=True),
            )
        self.feature_compute = self._global_feature_compute

        self.head = CompositeField4(meta, self.reduced_features * 2)

    def forward(self, *args):
        x = args[0]

        # Batches that are not intended for tracking loss might have an
        # odd number of images (or only 1 image).
        # In that case, simply do not execute this head as the result should
        # never be used.
        if len(x) % 2 == 1:
            return None

        x = self.feature_reduction(x)

        group_length = 2 if self.training else self.tracking_pose_length
        primary = x[::group_length]
        others = [x[i::group_length] for i in range(1, group_length)]

        x = torch.stack([torch.cat([primary, o], dim=1) for o in others], dim=1)
        x_shape = x.size()
        x = torch.reshape(x, [x_shape[0] * x_shape[1]] + list(x_shape[2:]))

        x = self.feature_compute(x)

        x = self.head(x)

        if self.tracking_pose_length != 2:
            # TODO need to stack group from batch dim in feature dim and adjust
            # meta info
            raise NotImplementedError

        return x
