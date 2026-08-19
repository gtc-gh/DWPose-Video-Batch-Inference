import cv2
import torch
import numpy as np
from . import util
from .wholebody import Wholebody


def draw_pose(pose, H, W, ori_img, bbox):
    bodies = pose['bodies']
    faces = pose['faces']
    foot = pose["foot"]
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']

    # canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = ori_img.copy()

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)
    canvas = util.draw_footpose(canvas, foot)
    # canvas = util.draw_facepose(canvas, faces)
    x1, y1, x2, y2 = map(int, bbox[0])
    canvas = cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return canvas


class DWProcessor:
    def __init__(self):

        self.pose_estimation = Wholebody()

    def __call__(self, video_imgs, candidate_, subset_, bbox_list, vis_res):
        imgs = video_imgs.copy()
        N, H, W, C = np.array(imgs).shape

        canvases = []
        pose_seq_ = []
        body_seq = []
        body_score = []
        hands_seq = []
        hands_score = []
        foot_seq = []
        foot_score = []
        faces_seq = []
        faces_score = []

        for candidate_i in range(N):
            candidate = candidate_[candidate_i]
            subset = subset_[candidate_i]
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body_seq.append(body[0].copy())
            body = body.reshape(nums * 18, locs)  # 18, 2
            score = subset[:, :18]
            body_score.append(score[0].copy())
            # print(np.array(body_score).shape)  # (N, 18)

            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.01:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.01
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]
            foot_seq.append(foot.copy())
            foot_score.append(subset[:, 18:24])

            faces = candidate[:, 24:92]
            faces_seq.append(faces.copy())
            faces_score.append(subset[:, 24:92])

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])
            hands_seq.append(candidate[:, 92:])
            hands_score.append(subset[:, 92:])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces, foot=foot)

            canvas = []
            if vis_res:
                canvas = draw_pose(pose, H, W, video_imgs[candidate_i], bbox_list[candidate_i])
            else:
                canvas = video_imgs[candidate_i].copy()
            canvases.append(canvas)


        # process body key points and their scores
        # print(f"body seq shape: {np.array(body_seq).shape}, score shape: {np.array(body_score).shape}")
        # body seq shape: (N, 18, 2), score shape: (N, 18)
        yolo_pose_map = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
        for i in range(N):
            pose_seq_i = []
            pose_seq_i.extend([0, 0, bbox_list[i][0][0], bbox_list[i][0][1],
                               bbox_list[i][0][2], bbox_list[i][0][3], 1.0])
            # add body
            for j in yolo_pose_map:
                if j == 1:
                    continue
                pose_seq_i.extend([body_seq[i][j][0], body_seq[i][j][1], body_score[i][j]])
            # add foot
            for f1, f2 in zip(foot_seq[i][0], foot_score[i][0]):
                pose_seq_i.extend([f1[0], f1[1], f2])
            # add faces
            for f1, f2 in zip(faces_seq[i][0], faces_score[i][0]):
                pose_seq_i.extend([f1[0], f1[1], f2])
            # add hands
            for h1, h2 in zip(hands_seq[i][0], hands_score[i][0]):
                pose_seq_i.extend([h1[0], h1[1], h2])

            pose_seq_.append(pose_seq_i)
        return canvases, np.array(pose_seq_)
