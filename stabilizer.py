import cv2
import numpy as np
import glob

last_T = np.zeros((2, 3))


class Trajectory:
    def __init__(self, x, y, a):
        self.x = x
        self.y = y
        self.a = a

    def __add__(self, other):
        return Trajectory(self.x + other.x, self.y + other.y, self.a + other.a)

    def __sub__(self, other):
        return Trajectory(self.x - other.x, self.y - other.y, self.a - other.a)

    def __mul__(self, other):
        return Trajectory(self.x * other.x, self.y * other.y, self.a * other.a)

    def __truediv__(self, other):
        return Trajectory(self.x / other.x, self.y / other.y, self.a / other.a)

    def __eq__(self, other):
        self.x = other.x
        self.y = other.y
        self.a = other.a
        return self


X = Trajectory(0, 0, 0)
P = Trajectory(1, 1, 1)
Q = Trajectory(0.1, 0.1, 0.1)
R = Trajectory(0.1, 0.1, 0.1)


def calculate_optical_flow(prev_gray, cur_gray, prev_corners):
    cur_corners, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, prev_corners, None)
    prev_corners_filtered = []
    cur_corners_filtered = []

    for i in range(len(status)):
        if status[i]:
            prev_corners_filtered.append(prev_corners[i])
            cur_corners_filtered.append(cur_corners[i])

    return np.array(prev_corners_filtered), np.array(cur_corners_filtered)


def estimate_affine_matrix(prev_corners, cur_corners):
    inlier_mask = np.zeros(len(prev_corners), dtype=bool)
    T, inlier_mask = cv2.estimateAffinePartial2D(
        prev_corners, cur_corners)

    if T is None:
        T = last_T.copy()

    last_T[...] = T

    dx = T[0, 2]
    dy = T[1, 2]
    da = np.arctan2(T[1, 0], T[0, 0])

    T_ = T.copy()
    T_[0, 2] = -dx
    T_[1, 2] = -dy

    T_[0, 0] = 1
    T_[0, 1] = 0
    T_[1, 0] = 0
    T_[1, 1] = 1

    return T, T_


def calc_stabli(image_files):
    global X, P, Q, R
    # 첫 번째 이미지를 이전 프레임으로 설정
    prev_frame = cv2.imread(image_files[0])
    prev_frame = cv2.resize(prev_frame, (1920, 1080))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    prev_corners = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

    # 이미지들을 순차적으로 처리
    for i in range(1, len(image_files)):
        # 현재 프레임
        cur_frame = cv2.imread(image_files[i])
        cur_frame = cv2.resize(cur_frame, (1920, 1080))
        cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

        # 광학 흐름 계산
        prev_corners, cur_corners = calculate_optical_flow(
            prev_gray, cur_gray, prev_corners)

        # 변환 매트릭스 추정
        T, T_ = estimate_affine_matrix(prev_corners, cur_corners)

        dx = T[0, 2]
        dy = T[1, 2]

        x = X.x + dx
        y = X.y + dy
        a = X.a
        z = Trajectory(x, y, a)

        if i == 1:
            X = Trajectory(0, 0, 0)
            P = Trajectory(1, 1, 1)
        else:
            X_ = X
            P_ = P + Q
            K = P_ / (P_ + R)
            X = X_ + K * (z - X_)
            P = (Trajectory(1, 1, 1) - K) * P_

        diff_x = (X.x - x)
        diff_y = (X.y - y)
        diff_a = (X.a - a)

        T_ = T.copy()
        T_[0, 2] = diff_x
        T_[1, 2] = diff_y

        T_[0, 0] = 1
        T_[0, 1] = 0
        T_[1, 0] = 0
        T_[1, 1] = 1

        if len(cur_corners) < 10 or abs(dx) > 10 or abs(dy) > 10:
            cur2 = cur_frame.copy()
            print(len(cur_corners), dx, dy)
            print('dx, dy too big')
        else:
            cur2 = cv2.warpAffine(cur_frame, T_, cur_frame.shape[:2])

        # 결과 이미지 출력
        cv2.imshow("Current Frame", cur_frame)
        cv2.imshow("Transformed Frame", cur2)
        cv2.waitKey(10)

        prev_gray = cur_gray.copy()
        prev_corners = cur_corners.copy()


cv2.destroyAllWindows()


image_files = glob.glob('frame/*.png')

calc_stabli(image_files)
