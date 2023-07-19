Reference: `basicvsr_plusplus_c64n7_8x1_600k_reds4.py` at MMEditing.

Due to the GPU memory limitation, we restrict the sample length to 7 for both training and testing at MFQEv2 dataset. Consequently, each sequence will be divided into several 7-frame sub-sequences. It is worth noting that the quality of border frames within each sub-sequence may be adversely affected due to this subdivision.
