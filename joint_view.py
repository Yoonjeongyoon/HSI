import numpy as np

data = np.load("sub16_clothesstand_000.npz")

# 데이터 확인
print(data.files)
print(data['seq_name'])             # 시퀀스 이름 (optional)
print(data['global_jpos'].shape) 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# SMPL-X 24-joint skeleton 연결 정보
EDGES = [
    (0, 1), (1, 4), (4, 7),         # left leg
    (0, 2), (2, 5), (5, 8),         # right leg
    (0, 3), (3, 6), (6, 9),         # spine
    (9, 12), (12, 15), (15, 18),    # left arm
    (9, 13), (13, 16), (16, 19),    # right arm
    (9, 10), (10, 11),              # neck → head
]

def view_with_slider(jpos_seq, edges=EDGES):
    T = jpos_seq.shape[0]

    # 중심 정렬 (pelvis 기준)
    jpos_seq_centered = jpos_seq - jpos_seq[:, 0:1, :]

    # 시각화 세팅
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    min_xyz = jpos_seq_centered.min(axis=(0, 1))
    max_xyz = jpos_seq_centered.max(axis=(0, 1))
    ax.set_xlim(min_xyz[0], max_xyz[0])
    ax.set_ylim(min_xyz[1], max_xyz[1])
    ax.set_zlim(min_xyz[2], max_xyz[2])
    ax.set_box_aspect([1, 1, 1])

    # 초기 프레임
    frame = 0
    joints = jpos_seq_centered[frame]
    scatter = ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', s=20)

    lines = []
    for i, j in edges:
        l, = ax.plot(
            [joints[i, 0], joints[j, 0]],
            [joints[i, 1], joints[j, 1]],
            [joints[i, 2], joints[j, 2]],
            c='b'
        )
        lines.append(l)

    # 슬라이더 생성 (하단)
    axframe = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(axframe, 'Frame', 0, T - 1, valinit=0, valstep=1)

    # 슬라이더 콜백 함수
    def update(val):
        f = int(slider.val)
        joints = jpos_seq_centered[f]
        scatter._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])
        for k, (i, j) in enumerate(edges):
            lines[k].set_data([joints[i, 0], joints[j, 0]], [joints[i, 1], joints[j, 1]])
            lines[k].set_3d_properties([joints[i, 2], joints[j, 2]])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

jpos_seq = data['global_jpos']  # shape: (T, 24, 3)

view_with_slider(jpos_seq)
