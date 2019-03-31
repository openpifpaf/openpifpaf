"""Head networks."""

import torch


def dequad(t):
    n_out_features = t.shape[1] // 4
    o = torch.zeros(
        (t.shape[0], n_out_features, t.shape[2] * 2 - 1, t.shape[3] * 2 - 1),
        device=t.device,
    )
    o[:, :, 0::2, 0::2] = t[:, 0:n_out_features]
    o[:, :, 1::2, 0::2] = t[:, n_out_features:n_out_features * 2, :-1, :]
    o[:, :, 0::2, 1::2] = t[:, n_out_features * 2:n_out_features * 3, :, :-1]
    o[:, :, 1::2, 1::2] = t[:, n_out_features * 3:n_out_features * 4, :-1, :-1]
    return o


class PartIntensityFields(torch.nn.Module):
    """Fields for parts."""

    def __init__(self, n_parts, in_features, shortname=None,
                 dropout_p=0.0, quad=0, with_scale=False):
        super(PartIntensityFields, self).__init__()
        self.shortname = shortname or 'pif{}{}'.format(
            's' if with_scale else '', n_parts)
        self.dilation = 1.0

        self.dropout = torch.nn.Dropout2d(p=dropout_p)
        self._quad = quad

        self.class_conv = torch.nn.Conv2d(in_features, n_parts * (4 ** self._quad), 1)
        self.reg_conv = torch.nn.Conv2d(in_features, n_parts * 2 * (4 ** self._quad), 1)
        torch.nn.init.normal_(self.reg_conv.weight, std=0.01)
        torch.nn.init.normal_(self.reg_conv.bias, std=0.01)
        self.scale_conv = None
        if with_scale:
            self.scale_conv = torch.nn.Conv2d(in_features, n_parts * (4 ** self._quad), 1)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)

        class_x = self.class_conv(x)
        class_x = torch.sigmoid(class_x)

        reg_x = self.reg_conv(x)
        reg_x = reg_x * self.dilation
        scale_x = None
        if self.scale_conv is not None:
            scale_x = self.scale_conv(x)
            scale_x = torch.nn.functional.relu(scale_x)

        for _ in range(self._quad):
            class_x = dequad(class_x)
            reg_x = dequad(reg_x)

            if scale_x is not None:
                scale_x = dequad(scale_x)

        reg_x = reg_x.reshape(reg_x.shape[0],
                              reg_x.shape[1] // 2,
                              2,
                              reg_x.shape[2],
                              reg_x.shape[3])

        out = [class_x, reg_x]
        if self.scale_conv is not None:
            out.append(scale_x)
        return out


class PartAssociationFields(torch.nn.Module):
    """Fields for associations."""

    def __init__(self, n_associations, in_features, shortname=None,
                 dropout_p=0.0, quad=0, predict_spread=False):
        super(PartAssociationFields, self).__init__()
        self.shortname = shortname or 'paf{}{}'.format(
            's' if predict_spread else '',
            n_associations,
        )
        self.dilation = 1.0

        self.dropout = torch.nn.Dropout2d(p=dropout_p)
        self._quad = quad

        out_features = n_associations * (4 ** self._quad)
        self.class_conv = torch.nn.Conv2d(in_features, out_features, 1)

        self.reg1_conv = torch.nn.Conv2d(in_features, 2 * out_features, 1)
        torch.nn.init.normal_(self.reg1_conv.weight, std=0.01)
        torch.nn.init.normal_(self.reg1_conv.bias, std=0.01)

        self.reg2_conv = torch.nn.Conv2d(in_features, 2 * out_features, 1)
        torch.nn.init.normal_(self.reg2_conv.weight, std=0.01)
        torch.nn.init.normal_(self.reg2_conv.bias, std=0.01)

        self.reg1_spread = None
        self.reg2_spread = None
        if predict_spread:
            self.reg1_spread = torch.nn.Conv2d(in_features, out_features, 1)
            self.reg2_spread = torch.nn.Conv2d(in_features, out_features, 1)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)

        # classification
        class_x = self.class_conv(x)
        class_x = torch.sigmoid(class_x)

        # regressions
        reg1_x = self.reg1_conv(x)
        reg1_x = reg1_x * self.dilation
        reg2_x = self.reg2_conv(x)
        reg2_x = reg2_x * self.dilation

        reg1_spread_x = None
        if self.reg1_spread is not None:
            reg1_spread_x = torch.clamp(self.reg1_spread(x), -3, 5)
        reg2_spread_x = None
        if self.reg2_spread is not None:
            reg2_spread_x = torch.clamp(self.reg2_spread(x), -3, 5)

        for _ in range(self._quad):
            class_x = dequad(class_x)
            reg1_x = dequad(reg1_x)
            reg2_x = dequad(reg2_x)

            if reg1_spread_x is not None:
                reg1_spread_x = dequad(reg1_spread_x)
            if reg2_spread_x is not None:
                reg2_spread_x = dequad(reg2_spread_x)

        reg1_x = reg1_x.reshape(reg1_x.shape[0],
                                reg1_x.shape[1] // 2,
                                2,
                                reg1_x.shape[2],
                                reg1_x.shape[3])
        reg2_x = reg2_x.reshape(reg2_x.shape[0],
                                reg2_x.shape[1] // 2,
                                2,
                                reg2_x.shape[2],
                                reg2_x.shape[3])

        out = [class_x, reg1_x, reg2_x]
        if reg1_spread_x is not None:
            out.append(reg1_spread_x)
        if reg2_spread_x is not None:
            out.append(reg2_spread_x)
        return out


class NPartAssociationFields(torch.nn.Module):
    """Fields for associations."""

    def __init__(self, n_associations, in_features, shortname=None,
                 dropout_p=0.0, quad=0, predict_spread=False, n_vectors=2):
        super(NPartAssociationFields, self).__init__()
        self.shortname = shortname or 'paf{}{}n{}'.format(
            's' if predict_spread else '',
            n_associations,
            n_vectors,
        )
        self.dilation = 1.0

        self.dropout = torch.nn.Dropout2d(p=dropout_p)
        self._quad = quad

        out_features = n_associations * (4 ** self._quad)
        self.class_conv = torch.nn.Conv2d(in_features, out_features, 1)

        self.reg_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, 2 * out_features, 1)
            for _ in range(n_vectors)
        ])
        for reg_conv in self.reg_convs:
            torch.nn.init.normal_(reg_conv.weight, std=0.01)
            torch.nn.init.normal_(reg_conv.bias, std=0.01)

        self.reg_spreads = None
        if predict_spread:
            self.reg_spreads = torch.nn.ModuleList([
                torch.nn.Conv2d(in_features, out_features, 1)
                for _ in self.reg_convs
            ])

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)

        # classification
        class_x = self.class_conv(x)
        class_x = torch.sigmoid(class_x)

        # regressions
        regs_x = [reg_conv(x) * self.dilation for reg_conv in self.reg_convs]

        regs_x_spread = None
        if self.reg_spreads is not None:
            regs_x_spread = [torch.clamp(reg_spread(x), -3, 5)
                             for reg_spread in self.reg_spreads]

        for _ in range(self._quad):
            class_x = dequad(class_x)
            regs_x = [dequad(reg_x) for reg_x in regs_x]

            if regs_x_spread is not None:
                regs_x_spread = [dequad(reg_x_spread) for reg_x_spread in regs_x_spread]

        regs_x = [
            reg_x.reshape(reg_x.shape[0],
                          reg_x.shape[1] // 2,
                          2,
                          reg_x.shape[2],
                          reg_x.shape[3])
            for reg_x in regs_x
        ]

        out = [class_x] + regs_x
        if regs_x_spread is not None:
            out += regs_x_spread
        return out


class CompositeField(torch.nn.Module):
    def __init__(self, n, in_features, shortname=None,
                 dropout_p=0.0, quad=0,
                 n_confidences=1, n_vectors=0, n_scales=0,
                 kernel_size=1, padding=0, dilation=1):
        super(CompositeField, self).__init__()
        self.shortname = shortname or 'cf{}c{}v{}s{}'.format(
            n,
            n_confidences,
            n_vectors,
            n_scales,
        )
        self.dilation = 1.0
        self.apply_class_sigmoid = True

        self.dropout = torch.nn.Dropout2d(p=dropout_p)
        self._quad = quad

        # classification
        out_features = n * (4 ** self._quad)
        self.class_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features, kernel_size,
                            padding=padding, dilation=dilation)
            for _ in range(n_confidences)
        ])

        # regression
        self.reg_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, 2 * out_features, kernel_size,
                            padding=padding, dilation=dilation)
            for _ in range(n_vectors)
        ])
        self.reg_spreads = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features, kernel_size,
                            padding=padding, dilation=dilation)
            for _ in self.reg_convs
        ])

        # scale
        self.scale_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features, kernel_size,
                            padding=padding, dilation=dilation)
            for _ in range(n_scales)
        ])

        # dequad
        self.dequad_op = torch.nn.PixelShuffle(2)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)

        # classification
        classes_x = [class_conv(x) for class_conv in self.class_convs]
        if self.apply_class_sigmoid:
            classes_x = [torch.sigmoid(class_x) for class_x in classes_x]

        # regressions
        regs_x = [reg_conv(x) * self.dilation for reg_conv in self.reg_convs]
        regs_x_spread = [reg_spread(x) for reg_spread in self.reg_spreads]

        # scale
        scales_x = [scale_conv(x) for scale_conv in self.scale_convs]
        scales_x = [torch.nn.functional.relu(scale_x) for scale_x in scales_x]

        for _ in range(self._quad):
            classes_x = [self.dequad_op(class_x)[:, :, :-1, :-1]
                         for class_x in classes_x]
            regs_x = [self.dequad_op(reg_x)[:, :, :-1, :-1]
                      for reg_x in regs_x]
            regs_x_spread = [self.dequad_op(reg_x_spread)[:, :, :-1, :-1]
                             for reg_x_spread in regs_x_spread]
            scales_x = [self.dequad_op(scale_x)[:, :, :-1, :-1]
                        for scale_x in scales_x]

        regs_x = [
            reg_x.reshape(reg_x.shape[0],
                          reg_x.shape[1] // 2,
                          2,
                          reg_x.shape[2],
                          reg_x.shape[3])
            for reg_x in regs_x
        ]

        return classes_x + regs_x + regs_x_spread + scales_x
