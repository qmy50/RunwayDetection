# row anchors are a series of pre-defined coordinates in image height to detect lanes
# the row anchors are defined according to the evaluation protocol of CULane and Tusimple
# since our method will resize the image to 288x800 for training, the row anchors are defined with the height of 288
# you can modify these row anchors according to your training image resolution

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]

# 1920 * 1080的图像，行锚点数量是56，行锚点坐标是从0.2到1.0等距划分的56个点乘以1080得到的整数坐标





tusimple_row_anchor = [96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 
                        174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240, 246, 
                        252, 258, 264, 270, 276, 282, 288, 294, 300, 306, 312, 318, 324, 330, 
                        336, 342, 348, 354, 360, 366, 372, 378, 384, 390, 396, 402, 408, 414, 420, 426]


culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

culane_col_anchor = [0.,  20.,  40.,  60.,  80., 100., 120., 140., 160., 180., 200.,
                    220., 240., 260., 280., 300., 320., 340., 360., 380., 400., 420.,
                    440., 460., 480., 500., 520., 540., 560., 580., 600., 620., 640.,
                    660., 680., 700., 720., 740., 760., 780., 800.]

tusimple_col_anchor = culane_col_anchor
