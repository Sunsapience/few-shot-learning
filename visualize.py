import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox


def visualize(embed, x_test):
    # 孪生网络编码后为2维向量，表示一个点的二位坐标
    # embed : 孪生网络中编码后的欧式距离 [None,2]
    # x_test: 测试图片 [None,28,28]

    feat = embed
    ax_min = np.min(embed,0)  # 坐标轴最小值
    ax_max = np.max(embed,0)
    ax_dist_sq = np.sum((ax_max-ax_min)**2)

    plt.figure()
    ax = plt.subplot(111)
    shown_images = np.array([[1., 1.]])
    for i in range(feat.shape[0]):
        dist = np.sum((feat[i] - shown_images)**2, 1)
        if np.min(dist) < 3e-4*ax_dist_sq:   # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [feat[i]]]
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(x_test[i], zoom=0.6),#, cmap=plt.cm.gray_r),
            xy=feat[i], frameon=False
        )
        ax.add_artist(imagebox)

    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    # plt.xticks([]), plt.yticks([])
    # plt.title('Embedding from the last layer of the network')
    plt.show()