# RyanXingQL, 2022
from ..builder import build_loss
from ..registry import MODELS
from .basic_restorer_qe import BasicRestorerQE


@MODELS.register_module()
class FreqLossRestorer(BasicRestorerQE):
    """Support frequency loss.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """
    def __init__(self,
                 generator,
                 pixel_loss,
                 freq_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator=generator,
                         pixel_loss=pixel_loss,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         pretrained=pretrained)

        # loss
        self.freq_loss = build_loss(freq_loss)

    def forward_train(self, lq, gt):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
        """
        losses = dict()
        output = self.generator(lq)
        losses['loss_pix'] = self.pixel_loss(output, gt)
        losses['loss_freq'] = self.freq_loss(output, gt)
        outputs = dict(losses=losses,
                       num_samples=len(gt.data),
                       results=dict(lq=lq.cpu(),
                                    gt=gt.cpu(),
                                    output=output.cpu()))
        return outputs
