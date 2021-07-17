from torch import nn
import torch
class MultiScaleSegmentation(nn.Module):
    #implentation of https://arxiv.org/pdf/2005.10821v1.pdf
    def __init__(self, model, scales:list, output_all_stages = False):
        super().__init__()
        self.model = model
        self.scales = scales
        self.output_all_stages = output_all_stages
        num_scales = len(scales)
        out_channels = self.model.encoder._out_channels[-1]


        self.scale_attn = nn.Sequential(
            nn.Conv2d(out_channels * num_scales, 512, kernel_size=3,
                      padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_scales, kernel_size=1, padding=1,
                      bias=False))
        #self.classification_head = ClassificationHead()
    def scale_as(self, x,y):
        '''
        Scale x to the size of y
        '''
        y_size = y.size(2), y.size(3)
        x_scaled = torch.nn.functional.interpolate(
            x, size=y_size, mode='bilinear',
            align_corners=True)
        return x_scaled
    def ResizeX(self, x, scale_factor):
        return torch.nn.functional.interpolate(
            x, scale_factor=scale_factor, mode='bilinear',
            align_corners=True, recompute_scale_factor=True)
        
    def forward(self, x):
        assert 1.0 in self.scales
        x_1x = x
        ps = {}
        ps[1.0], feats_1x = self.model(x, False)
        #feats_1x = self.classification_head(feats_1x)
        concat_feats = [feats_1x]
        
        for scale in self.scales:
            if scale == 1.0:
                continue
            resized_x = self.ResizeX(x_1x, scale)
            p, feats = self.model(resized_x, False)
            #feats = self.classification_head(feats)
            ps[scale] = self.scale_as(p, x_1x)
            feats = self.scale_as(feats, feats_1x)
            concat_feats.append(feats)
            
        concat_feats = torch.cat(concat_feats, 1)
        attn_tensor = self.scale_attn(concat_feats)
        output = None

        for idx, scale in enumerate(self.scales):
            attn = attn_tensor[:, idx:idx+1, :, :]
            attn_1x_scale = self.scale_as(attn, x_1x)
            if output is None:
                output = ps[scale] * attn_1x_scale
            else:
                output += ps[scale] * attn_1x_scale


        return output

def fmt_scale(prefix, scale):
    """
    format scale name
    :prefix: a string that is the beginning of the field name
    :scale: a scale value (0.25, 0.5, 1.0, 2.0)
    """

    scale_str = str(float(scale))
    scale_str.replace('.', '')
    return f'{prefix}_{scale_str}x'


class HieraricalMultiScale(nn.Module):
    def __init__(self, model, scales:list):
        super().__init__()
        self.model = model
        self.scales = scales
        out_channels = self.model.encoder._out_channels[-1]
        self.attn = nn.Sequential(
                nn.Conv2d(out_channels, 512, kernel_size=3, padding=1, bias=False),
                nn.Norm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                nn.Norm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 1, kernel_size=1, bias=False),
                nn.Sigmoid())

    def scale_as(self, x,y):
        '''
        Scale x to the size of y
        '''
        y_size = y.size(2), y.size(3)
        x_scaled = torch.nn.functional.interpolate(
            x, size=y_size, mode='bilinear',
            align_corners=True)
        return x_scaled
    def ResizeX(self, x, scale_factor):
        return torch.nn.functional.interpolate(
            x, scale_factor=scale_factor, mode='bilinear',
            align_corners=True, recompute_scale_factor=True)
        
    def forward(self, x):
        assert 1.0 in self.scales
        x_1x = x

        pred = None
        output_dict = {}

        for s in self.scales:
            x = self.ResizeX(x_1x, s)
            
            p, attn = self.model(x, True)
            scale_attn = self.scale_attn(attn)
            output_dict[fmt_scale('pred', s)] = p
            if s != 2.0:
                output_dict[fmt_scale('attn', s)] = attn

            if pred is None:
                pred = p
            elif s >= 1.0:
                # downscale previous
                pred = self.scale_as(pred, p)
                pred = attn * p + (1 - attn) * pred
            else:
                # upscale current
                p = attn * p
                p = self.scale_as(p, pred)
                attn = self.scale_as(attn, pred)
                pred = p + (1 - attn) * pred
        return pred