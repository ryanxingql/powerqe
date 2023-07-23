Reference: `basicvsr_plusplus_c64n7_8x1_600k_reds4.py` at MMEditing.

Due to the GPU memory limitation, we restrict the sample length to 7 for both training and testing at MFQEv2 dataset. Consequently, each sequence will be divided into several 7-frame sub-sequences. It is worth noting that the quality of border frames within each sub-sequence may be adversely affected due to this subdivision.

The width and height of input frames should be larger than 256 as requested by BasicVSR++. However, some videos in the MFQEv2 dataset do not meet this requirement. As a result, we employ padding in the test config.
