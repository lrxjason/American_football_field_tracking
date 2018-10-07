import cv2
import numpy as np
from skimage.measure import compare_ssim

width = 960
height = 720


def grayscale(image):
    """
    :param image: RGB image
    :return: grayscale image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_corner_point(box):
    """
    :param box: location of bounding box (top_left_x, top_left_y, width, height)
    :return: 4 corner points of the box
    """
    point1 = [box[0], box[1]]
    point2 = [box[0] + box[2], box[1]]
    point3 = [box[0], box[1] + box[3]]
    point4 = [box[0] + box[2], box[1] + box[3]]
    return np.array([point1, point2, point3, point4])


def homotransform_on_point(point, H):
    """
    :param point: point pixel location in image1
    :param H: homography matrix
    :return: point pixel location in image2
    """
    p = np.array((point[0], point[1], 1)).reshape((3, 1))
    temp_p = H.dot(p)
    total = np.sum(temp_p, 1)
    px = int(round(total[0] / total[2]))
    py = int(round(total[1] / total[2]))
    return px, py


def homotransform_on_points(points, H):
    """
    :param points: points pixel location in image1
    :param H: homography matrix
    :return: points pixel location in image2
    """
    h_p = []
    for i in range(len(points)):
        p = np.array((points[i][0], points[i][1], 1)).reshape((3, 1))
        temp_p = H.dot(p)
        total = np.sum(temp_p, 1)
        px = int(round(total[0] / total[2]))
        py = int(round(total[1] / total[2]))
        h_p.append([px, py])
    return np.array(h_p)


def point_to_box(point, size):
    """
    :param point: feature point
    :param size: bounding box side length
    :return: location of bounding box (top_left_x, top_left_y, width, height)
    """
    box = (round(point[0] - size / 2), round(point[1] - size / 2), size, size)
    return box


def pixle_distance(point1, point2):
    """
    :param point1: point1 pixel location
    :param point2: point2 pixel location
    :return: pixel distance betweent point1 and point2
    """
    d = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return d


def mean_position_of_points(points):
    """
    :param points: corner points list
    :return: mean position of these points
    """
    x = 0
    y = 0
    for i in range(len(points)):
        x = x + points[i][0][0]
        y = y + points[i][0][1]
    mean_x = x / len(points)
    mean_y = y / len(points)
    return int(round(mean_x)), int(round(mean_y))


def center_of_box(box):
    """
    :param box: location of bounding box (top_left_x, top_left_y, width, height)
    :return: center point pixel location
    """
    x = box[0] + box[2] / 2
    y = box[1] + box[3] / 2
    return int(round(x)), int(round(y))


def check_tracked(tracked_list):
    """
    :param tracked_list: all intersection are visible on frame
    :return: number of features on each hash lines
    """
    top_hash_number_tracking = 0
    bottom_hash_number_tracking = 0
    for i in range(len(tracked_list)):
        if tracked_list[i] == 1:
            # if even
            if i % 2 == 0:
                top_hash_number_tracking += 1
            # if odd
            else:
                bottom_hash_number_tracking += 1
    return top_hash_number_tracking, bottom_hash_number_tracking


class tracking:
    """
    frame:  current frame using for draw information
    boxes: tracking boxes
    tracked_list: all intersection are visible on frame
    feature_points: good points for tracking
    trackers: opencv tracking API (KCF)
    states: states of the trackers
    """

    def __init__(self, frame, H_frame_to_court, box, feature_list, parameter):
        """
        :param frame:  current frame using for draw information
        :param H_frame_to_court: homography matrix
        :param box: searching box
        :param feature_list: all intersection are visible on frame
        :param parameter: parametar for feature detection
        """
        self.frame = frame
        self.boxes = box
        self.tracked_list = feature_list
        self.feature_points = []
        self.trackers = []
        self.states = []
        image = frame.copy()
        crop_copy = frame.copy()
        # all yard lines and hash lines intersections on court
        self.total_intersections = len(self.tracked_list)

        # pixel distance between two hash line on frame
        line1 = homotransform_on_point([1000, 370], H_frame_to_court)
        line2 = homotransform_on_point([1000, 590], H_frame_to_court)
        self.pixel_distance_initial = pixle_distance(line1, line2)

        # loop for all intersections
        for i in range(self.total_intersections):
            # if intersection on frame
            if self.tracked_list[i] == 1:
                # plot the searching box
                p1 = (int(self.boxes[i][0]), int(self.boxes[i][1]))
                p2 = (int(self.boxes[i][0] + self.boxes[i][2]), int(self.boxes[i][1] + self.boxes[i][3]))
                cv2.rectangle(self.frame, p1, p2, (255, 0, 0), 2, 1)
                imCrop = image[int(self.boxes[i][1]):int(self.boxes[i][1] + self.boxes[i][3]),
                         int(self.boxes[i][0]):int(self.boxes[i][0] + self.boxes[i][2])]
                gray = grayscale(imCrop)
                # try to detect the feature points on this searching area
                try:
                    p0 = cv2.goodFeaturesToTrack(gray, mask=None, **parameter.feature_params)
                    center_local = mean_position_of_points(p0)
                    # use the mean position as the tracking box center
                    center = (center_local[0] + self.boxes[i][0], center_local[1] + self.boxes[i][1])
                    # get the tracking box
                    box_temp = point_to_box(center, parameter.tracking_box_size)
                    # plot the tracking box
                    p1 = (int(box_temp[0]), int(box_temp[1]))
                    p2 = (int(box_temp[0] + box_temp[2]), int(box_temp[1] + box_temp[3]))
                    cv2.rectangle(self.frame, p1, p2, (255, 255, 255), 2, 1)
                    flag = 1
                except:
                    self.boxes[i] = []
                    self.tracked_list[i] = 0
                    self.feature_points.append([])
                    flag = 0

                # if get the tracking box
                if flag == 1:
                    # find the box image pattern on court
                    H_image = cv2.warpPerspective(crop_copy, H_frame_to_court, (2000, 965))
                    H_center = homotransform_on_point(center, H_frame_to_court)
                    H_box = point_to_box(H_center, parameter.tracking_box_size)
                    H_crop = H_image[int(H_box[1]):int(H_box[1] + H_box[3]), int(H_box[0]):int(H_box[0] + H_box[2])]
                    gray2 = grayscale(H_crop)
                    # compare the image with template image to select non-occlusion pattern
                    # (score, diff) = compare_ssim(gray2, template, full=True)
                    im_lr = np.fliplr(gray2)
                    im_up = np.flipud(gray2)
                    (score1, diff1) = compare_ssim(gray2, im_lr, full=True)
                    (score2, diff2) = compare_ssim(gray2, im_up, full=True)
                    (score3, diff3) = compare_ssim(im_lr, im_up, full=True)
                    score = score1 + score2 + score3
                    # print(score)
                    if score > 0.15:
                        self.feature_points.append(center)
                        box_temp = point_to_box(center, parameter.tracking_box_size)
                        # set the tracking box to class variable
                        self.boxes[i] = box_temp
                        # plot the final tracking box
                        p1 = (int(box_temp[0]), int(box_temp[1]))
                        p2 = (int(box_temp[0] + box_temp[2]), int(box_temp[1] + box_temp[3]))
                        cv2.rectangle(self.frame, p1, p2, (0, 0, 255), 2, 1)
                    else:
                        self.feature_points.append([])
                        self.tracked_list[i] = 0
                        self.boxes[i] = []
            else:
                self.feature_points.append([])

    def tracking_create(self):
        """
        :return: if intersection on frame create a tracker for this intersection
        """
        for features_on_frame in range(self.total_intersections):
            if self.tracked_list[features_on_frame] == 1:
                tracker = cv2.TrackerKCF_create()
                self.trackers.append(tracker)
            else:
                self.trackers.append([])

    def tracking_initial(self):
        """
        :return: if intersection on frame create a state for this intersection
        """
        for features_on_frame in range(self.total_intersections):
            if self.tracked_list[features_on_frame] == 1:
                state_temp = self.trackers[features_on_frame].init(self.frame, self.boxes[features_on_frame])
                self.states.append(state_temp)
            else:
                self.states.append([])

    def tracking_update(self, frame):
        """
        :param frame: update the frame image
        :return:  the state of this tracker
        """
        self.frame = frame
        for features_on_frame in range(self.total_intersections):
            if self.tracked_list[features_on_frame] == 1:
                self.states[features_on_frame], self.boxes[features_on_frame] = \
                    self.trackers[features_on_frame].update(self.frame)

    def tracking_check_state(self):
        """
        :return: if tracking box is tracked success update the box and return state as True
        """
        for i in range(self.total_intersections):
            if self.tracked_list[i] == 1:
                # Draw bounding boxes
                if self.states[i]:
                    # Tracking success
                    p1 = (int(self.boxes[i][0]), int(self.boxes[i][1]))
                    p2 = (int(self.boxes[i][0] + self.boxes[i][2]),
                          int(self.boxes[i][1] + self.boxes[i][3]))
                    cv2.rectangle(self.frame, p1, p2, (0, 0, 255), 2, 1)
                    self.feature_points[i] = center_of_box(self.boxes[i])
                else:
                    self.boxes[i] = []
                    self.tracked_list[i] = 0
                    self.feature_points[i] = []
                    # Tracking failure
                    cv2.putText(self.frame, "Tracking feature " + str(i) + " failure detected", (100, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    def tracking_redetection(self, frame, H_frame_to_court, box, feature_list, parameter):
        """
        :param frame:  current frame using for draw information
        :param H_frame_to_court: homography matrix
        :param box: searching box
        :param feature_list: all intersection are visible on frame
        :param parameter: parameter for feature detection
        """
        self.tracked_list = feature_list
        image = frame.copy()
        crop_copy = frame.copy()
        self.frame = frame
        for i in range(self.total_intersections):
            if self.tracked_list[i] == 1:
                p1 = (int(box[i][0]), int(box[i][1]))
                p2 = (int(box[i][0] + box[i][2]), int(box[i][1] + box[i][3]))
                cv2.rectangle(self.frame, p1, p2, (255, 0, 0), 2, 1)
                imCrop = image[int(box[i][1]):int(box[i][1] + box[i][3]),
                         int(box[i][0]):int(box[i][0] + box[i][2])]
                gray = grayscale(imCrop)
                try:
                    p0 = cv2.goodFeaturesToTrack(gray, mask=None, **parameter.feature_params)
                    center_local = mean_position_of_points(p0)
                    center = (center_local[0] + box[i][0], center_local[1] + box[i][1])
                    box_temp = point_to_box(center, parameter.tracking_box_size)
                    p1 = (int(box_temp[0]), int(box_temp[1]))
                    p2 = (int(box_temp[0] + box_temp[2]), int(box_temp[1] + box_temp[3]))
                    cv2.rectangle(self.frame, p1, p2, (255, 255, 255), 2, 1)
                    flag = 1
                except:
                    self.boxes[i] = []
                    self.tracked_list[i] = 0
                    self.states[i] = []
                    self.feature_points[i] = []
                    flag = 0

                if flag == 1:
                    H_image = cv2.warpPerspective(crop_copy, H_frame_to_court, (2000, 965))
                    H_center = homotransform_on_point(center, H_frame_to_court)
                    H_box = point_to_box(H_center, parameter.tracking_box_size)
                    H_crop = H_image[int(H_box[1]):int(H_box[1] + H_box[3]), int(H_box[0]):int(H_box[0] + H_box[2])]
                    gray2 = grayscale(H_crop)
                    # (score, diff) = compare_ssim(gray2, template, full=True)
                    im_lr = np.fliplr(gray2)
                    im_up = np.flipud(gray2)
                    (score1, diff1) = compare_ssim(gray2, im_lr, full=True)
                    (score2, diff2) = compare_ssim(gray2, im_up, full=True)
                    (score3, diff3) = compare_ssim(im_lr, im_up, full=True)
                    score = score1 + score2 + score3
                    # print(score)
                    if score > 0.15:
                        self.feature_points[i] = center
                        box_temp = point_to_box(center, parameter.tracking_box_size)
                        self.boxes[i] = box_temp
                        p1 = (int(box_temp[0]), int(box_temp[1]))
                        p2 = (int(box_temp[0] + box_temp[2]), int(box_temp[1] + box_temp[3]))
                        cv2.rectangle(self.frame, p1, p2, (0, 0, 255), 2, 1)
                        self.trackers[i] = cv2.TrackerKCF_create()
                        self.states[i] = self.trackers[i].init(image, self.boxes[i])
                    else:
                        self.boxes[i] = []
                        self.tracked_list[i] = 0
                        self.states[i] = []
                        self.feature_points[i] = []
            else:
                self.boxes[i] = []
                self.tracked_list[i] = 0
                self.states[i] = []
                self.feature_points[i] = []

    def tracking_interest_area(self, interest_area_on_court, H_frame_to_court):
        """
        :param interest_area_on_court: corner points of the interest are box
        :param H_frame_to_court: homography matrix
        :return: corner points of the interest area
        """
        project_points = homotransform_on_points(interest_area_on_court, np.linalg.inv(H_frame_to_court))
        # plot the tracking field on current frame
        cv2.line(self.frame, (project_points[0][0], project_points[0][1]),
                 (project_points[1][0], project_points[1][1]), (200, 0, 0), 2)
        cv2.line(self.frame, (project_points[1][0], project_points[1][1]),
                 (project_points[3][0], project_points[3][1]), (200, 0, 0), 2)
        cv2.line(self.frame, (project_points[2][0], project_points[2][1]),
                 (project_points[3][0], project_points[3][1]), (200, 0, 0), 2)
        cv2.line(self.frame, (project_points[2][0], project_points[2][1]),
                 (project_points[0][0], project_points[0][1]), (200, 0, 0), 2)
        return project_points


def video_imformation(folder_name, video_name, out_put_folder):
    # input video
    cap = cv2.VideoCapture(folder_name + video_name + '.mxf')
    # frame width
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame height
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # total frame in this video
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (width, height)
    # output format is mp4
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    # output video
    output_video = cv2.VideoWriter(out_put_folder + video_name + '.mp4', codec, 60, size)
    return cap, width, height, size, total_frame, output_video


def get_initial_H_matrix(cap, court):
    """
    :param cap: input video
    :param court:  court image
    :return: Homography matrix
     map the first frame to the court image
    """
    ret, frame = cap.read()
    frame_box1 = cv2.selectROI(frame, True)
    print(frame_box1)
    court_box1 = cv2.selectROI(court, True)
    print(court_box1)

    frame_box2 = cv2.selectROI(frame, True)
    print(frame_box2)
    court_box2 = cv2.selectROI(court, True)
    print(court_box2)

    frame_box3 = cv2.selectROI(frame, True)
    print(frame_box3)
    court_box3 = cv2.selectROI(court, True)
    print(court_box3)

    frame_box4 = cv2.selectROI(frame, True)
    print(frame_box4)
    court_box4 = cv2.selectROI(court, True)
    print(court_box4)

    frame_point1 = [int(frame_box1[0] + frame_box1[2] / 2), int(frame_box1[1] + frame_box1[3] / 2)]
    frame_point2 = [int(frame_box2[0] + frame_box2[2] / 2), int(frame_box2[1] + frame_box2[3] / 2)]
    frame_point3 = [int(frame_box3[0] + frame_box3[2] / 2), int(frame_box3[1] + frame_box3[3] / 2)]
    frame_point4 = [int(frame_box4[0] + frame_box4[2] / 2), int(frame_box4[1] + frame_box4[3] / 2)]

    court_point1 = [int(court_box1[0] + court_box1[2] / 2), int(court_box1[1] + court_box1[3] / 2)]
    court_point2 = [int(court_box2[0] + court_box2[2] / 2), int(court_box2[1] + court_box2[3] / 2)]
    court_point3 = [int(court_box3[0] + court_box3[2] / 2), int(court_box3[1] + court_box3[3] / 2)]
    court_point4 = [int(court_box4[0] + court_box4[2] / 2), int(court_box4[1] + court_box4[3] / 2)]

    pts1 = np.float32([frame_point1, frame_point2, frame_point3, frame_point4])
    pts2 = np.float32([court_point1, court_point2, court_point3, court_point4])



    # Define an initial bounding box for    Rudy Sanchez - CFB ALL 22 1
    # bbox1 = (131, 52, 93, 37)
    # bbox2 = (610, 22, 94, 43)
    # bbox3 = (467, 650, 118, 67)
    # bbox4 = (769, 630, 119, 67)
    #
    # h1 = [int(bbox1[0] + bbox1[2] / 2), int(bbox1[1] + bbox1[3] / 2)]
    # h2 = [int(bbox2[0] + bbox2[2] / 2), int(bbox2[1] + bbox2[3] / 2)]
    # h3 = [int(bbox3[0] + bbox3[2] / 2), int(bbox3[1] + bbox3[3] / 2)]
    # h4 = [int(bbox4[0] + bbox4[2] / 2), int(bbox4[1] + bbox4[3] / 2)]
    #
    # pts1 = np.float32([h1, h2, h3, h4])
    # pts2 = np.float32([[1000, 185], [1318, 185], [1159, 785], [1317, 785]])



    # Define an initial bounding box for    Rudy Sanchez - CFB ALL 22 2
    # bbox1 = (48, 123, 103, 55)
    # bbox2 = (768, 45, 106, 55)
    # bbox3 = (848, 429, 74, 54)
    # bbox4 = (146, 510, 68, 71)
    #
    # h1 = [int(bbox1[0] + bbox1[2] / 2), int(bbox1[1] + bbox1[3] / 2)]
    # h2 = [int(bbox2[0] + bbox2[2] / 2), int(bbox2[1] + bbox2[3] / 2)]
    # h3 = [int(bbox3[0] + bbox3[2] / 2), int(bbox3[1] + bbox3[3] / 2)]
    # h4 = [int(bbox4[0] + bbox4[2] / 2), int(bbox4[1] + bbox4[3] / 2)]
    #
    # court_box1 = (1102, 140, 111, 93)
    # court_box2 = (1571, 144, 125, 86)
    # court_box3 = (1532, 560, 42, 62)
    # court_box4 = (1137, 559, 40, 63)
    #
    # c1 = [int(court_box1[0] + court_box1[2] / 2), int(court_box1[1] + court_box1[3] / 2)]
    # c2 = [int(court_box2[0] + court_box2[2] / 2), int(court_box2[1] + court_box2[3] / 2)]
    # c3 = [int(court_box3[0] + court_box3[2] / 2), int(court_box3[1] + court_box3[3] / 2)]
    # c4 = [int(court_box4[0] + court_box4[2] / 2), int(court_box4[1] + court_box4[3] / 2)]
    #
    # pts1 = np.float32([h1, h2, h3, h4])
    # pts2 = np.float32([c1, c2, c3, c4])



    # # Define an initial bounding box for    Rudy Sanchez - NFL HIGH ENDZONE 1
    # bbox1 = (316, 635, 50, 61)
    # bbox2 = (649, 636, 37, 56)
    # bbox3 = (341, 34, 52, 55)
    # bbox4 = (592, 40, 51, 39)
    #
    # h1 = [int(bbox1[0] + bbox1[2] / 2), int(bbox1[1] + bbox1[3] / 2)]
    # h2 = [int(bbox2[0] + bbox2[2] / 2), int(bbox2[1] + bbox2[3] / 2)]
    # h3 = [int(bbox3[0] + bbox3[2] / 2), int(bbox3[1] + bbox3[3] / 2)]
    # h4 = [int(bbox4[0] + bbox4[2] / 2), int(bbox4[1] + bbox4[3] / 2)]
    #
    # court_box1 = (829, 353, 28, 38)
    # court_box2 = (820, 565, 45, 50)
    # court_box3 = (1380, 342, 36, 56)
    # court_box4 = (1370, 562, 54, 53)
    #
    # c1 = [int(court_box1[0] + court_box1[2] / 2), int(court_box1[1] + court_box1[3] / 2)]
    # c2 = [int(court_box2[0] + court_box2[2] / 2), int(court_box2[1] + court_box2[3] / 2)]
    # c3 = [int(court_box3[0] + court_box3[2] / 2), int(court_box3[1] + court_box3[3] / 2)]
    # c4 = [int(court_box4[0] + court_box4[2] / 2), int(court_box4[1] + court_box4[3] / 2)]
    #
    # pts1 = np.float32([h1, h2, h3, h4])
    # pts2 = np.float32([c1, c2, c3, c4])

    H_frame_to_court = cv2.getPerspectiveTransform(pts1, pts2)
    return H_frame_to_court


def project_hash_yard_intersection_box(H_frame_to_court, size, intersection_flag):
    """
    :param H_frame_to_court: current homography matrix (map frame to court)
    :param size: search features bounding box size
    :param intersection_flag:
            1 : intersection include the court boundary and hash lines
            2 : intersection just include hash lines
    :return:
            boxes
            the bounding box location on frame and feature index
            (top_left_x, top_left_y, width, height)
    """
    boxes = []
    intersection1 = ((56, 57), (56, 907),
                     (210, 57), (210, 907),
                     (291, 57), (291, 370), (291, 590), (291, 907),
                     (369, 57), (369, 370), (369, 590), (369, 907),
                     (449, 57), (449, 370), (449, 590), (449, 907),
                     (527, 57), (527, 370), (527, 590), (527, 907),
                     (607, 57), (607, 370), (607, 590), (607, 907),
                     (685, 57), (685, 370), (685, 590), (685, 907),
                     (765, 57), (765, 370), (765, 590), (765, 907),
                     (842, 57), (842, 370), (842, 590), (842, 907),
                     (923, 57), (923, 370), (923, 590), (923, 907),
                     (1000, 57), (1000, 370), (1000, 590), (1000, 907),
                     (1081, 57), (1081, 370), (1081, 590), (1081, 907),
                     (1158, 57), (1158, 370), (1158, 590), (1158, 907),
                     (1239, 57), (1239, 370), (1239, 590), (1239, 907),
                     (1316, 57), (1316, 370), (1316, 590), (1316, 907),
                     (1396, 57), (1396, 370), (1396, 590), (1396, 907),
                     (1474, 57), (1474, 370), (1474, 590), (1474, 907),
                     (1554, 57), (1554, 370), (1554, 590), (1554, 907),
                     (1632, 57), (1632, 370), (1632, 590), (1632, 907),
                     (1712, 57), (1712, 370), (1712, 590), (1712, 907),
                     (1790, 57), (1790, 907),
                     (1944, 57), (1944, 907))
    intersection2 = ((291, 370), (291, 590),
                     (369, 370), (369, 590),
                     (449, 370), (449, 590),
                     (527, 370), (527, 590),
                     (607, 370), (607, 590),
                     (685, 370), (685, 590),
                     (765, 370), (765, 590),
                     (842, 370), (842, 590),
                     (923, 370), (923, 590),
                     (1000, 370), (1000, 590),
                     (1081, 370), (1081, 590),
                     (1158, 370), (1158, 590),
                     (1239, 370), (1239, 590),
                     (1316, 370), (1316, 590),
                     (1396, 370), (1396, 590),
                     (1474, 370), (1474, 590),
                     (1554, 370), (1554, 590),
                     (1632, 370), (1632, 590),
                     (1712, 370), (1712, 590))
    if intersection_flag == 1:
        intersection = intersection1
    else:
        intersection = intersection2

    tracked_list = np.zeros(len(intersection))
    for i in range(len(intersection)):
        intersection_pixel_on_frame = homotransform_on_point(
            intersection[i], np.linalg.inv(H_frame_to_court))
        box = point_to_box((intersection_pixel_on_frame[0], intersection_pixel_on_frame[1]), size)
        # if the tersection box are on the frame, return it
        if box[0] > 0 and box[1] > 0 and box[0] + box[2] < width and box[1] + box[3] < height:
            tracked_list[i] = 1
            boxes.append(box)
        else:
            boxes.append([])
    return boxes, tracked_list


def count_searching_box(tracked_list):
    upper_hash = 0
    lower_hash = 0
    if len(tracked_list) < 40:  # only using hash line and yard line intersections
        for i in range(len(tracked_list)):
            if tracked_list[i] == 1:
                if i % 2 == 0:
                    upper_hash += 1
                else:
                    lower_hash += 1
    else:  # Also include boundary line and yard line intersections
        for i in range(len(tracked_list)):
            if tracked_list[i] == 1:
                if i % 4 == 1:
                    upper_hash += 1
                if i % 4 == 2:
                    lower_hash += 1
    upper_hash_thresh = int(round(upper_hash/2))
    lower_hash_thresh = int(round(lower_hash / 2))
    return upper_hash_thresh, lower_hash_thresh


class camera_parameter:
    def __init__(self, H_frame_to_court, boxes_search, tracked_list, initial_distance):
        """
        :param H_frame_to_court:  current homography matrix (map frame to court)
        :param pixel_distance_initial: pixel distance between two hash line on frame
        """
        self.initial = initial_distance
        self.count = 0
        self.start = 1
        self.state = True
        box_upper = []
        box_lower = []

        '''
        check camera view point
        '''
        if len(tracked_list) < 40: # only using hash line and yard line intersections
            for i in range(len(tracked_list)):
                if tracked_list[i] == 1:
                    if i % 2 == 0:
                        box_upper.append([boxes_search[i][0], boxes_search[i][1]])
                    else:
                        box_lower.append([boxes_search[i][0], boxes_search[i][1]])
            box_upper = np.std(box_upper, axis=0)  # find the stander deviation of searching box
            box_lower = np.std(box_lower, axis=0)
            box_std = box_upper + box_lower
        else:  # Also include boundary line and yard line intersections
            for i in range(len(tracked_list)):
                if tracked_list[i] == 1:
                    if i % 4 == 1:
                        box_upper.append([boxes_search[i][0], boxes_search[i][1]])
                    if i % 4 == 2:
                        box_lower.append([boxes_search[i][0], boxes_search[i][1]])
            box_upper = np.std(box_upper)
            box_lower = np.std(box_lower)
            box_std = box_upper + box_lower

        # if y deviation is larger than x means the camera is end view
        if box_std[1] > box_std[0]:
            self.view_type = 0
            self.P, self.T, self.Z = compute_PTZ_end(H_frame_to_court, initial_distance)
        else:  # camera is side view
            self.view_type = 1
            self.P, self.T, self.Z = compute_PTZ_side(H_frame_to_court, initial_distance)

    def state_check(self, H_frame_to_court, current_frame):
        """
        :param H_frame_to_court: current homography matrix (map frame to court)
        :param current_frame:  current frame number
        :return: camera PTZ state
        """
        self.state = True
        if self.view_type == 0:
            Pan_angle, tilt_angle, Zoom = compute_PTZ_end(H_frame_to_court, self.initial)
            thresh = 1
        else:
            Pan_angle, tilt_angle, Zoom = compute_PTZ_side(H_frame_to_court, self.initial)
            thresh = 0.5
        change = np.sum(np.abs(Pan_angle - self.P) + np.abs(tilt_angle - self.T) + np.abs(Zoom - self.Z))
        # if two continuous frame PTZ change too big, we consider it lose tracking
        if change > thresh:
            print(self.start)
            frame_elapse = current_frame - self.start
            if frame_elapse > 5:
                self.count = 1
                self.start = current_frame
            else:
                self.count += 1
        if self.count > 2:
            self.state = False
        self.P = Pan_angle
        self.T = tilt_angle
        self.Z = Zoom
        return self.state


def compute_PTZ_side(H_frame_to_court, pixel_distance_initial):
    """
      :param H_frame_to_court:  current homography matrix (map frame to court)
      :param pixel_distance_initial: pixel distance between two hash line on frame
      :return: camera PTZ value
      """
    '''
    Pan calculate
    '''
    corner1 = homotransform_on_point([0, 0], H_frame_to_court)
    corner2 = homotransform_on_point([width, 0], H_frame_to_court)
    corner3 = homotransform_on_point([0, height], H_frame_to_court)
    corner4 = homotransform_on_point([width, height], H_frame_to_court)
    center = homotransform_on_point([width / 2, height / 2], H_frame_to_court)
    slop_k = (corner2[1] - corner1[1]) / (corner2[0] - corner1[0])
    slop_k2 = (corner4[1] - corner3[1]) / (corner4[0] - corner3[0])
    Pan_angle = np.round(np.arctan((slop_k + slop_k2) / 2) * 180 / np.pi, 1)

    '''
    Tilt calculation
    '''
    # center to bottom boundary
    distance_per_pixel = 0.187  # 80/(910-482)
    distance = 80 + (482 - center[1]) * distance_per_pixel
    dis_to_camera = distance + 20
    tilt_angle = np.round(np.arctan(100 / dis_to_camera) * 180 / np.pi, 1)

    '''
    Zoom calculate
    '''
    line1 = homotransform_on_point([1000, 370], H_frame_to_court)
    line2 = homotransform_on_point([1000, 590], H_frame_to_court)
    pixel_distance_update = pixle_distance(line1, line2)
    Zoom = np.round(pixel_distance_initial / pixel_distance_update, 1)

    return Pan_angle, tilt_angle, Zoom


def compute_PTZ_end(H_frame_to_court, pixel_distance_initial):
    """
      :param H_frame_to_court:  current homography matrix (map frame to court)
      :param pixel_distance_initial: pixel distance between two hash line on frame
      :return: camera PTZ value
      """
    '''
    Pan calculate
    '''
    corner1 = homotransform_on_point([0, 0], H_frame_to_court)
    corner2 = homotransform_on_point([width, 0], H_frame_to_court)
    corner3 = homotransform_on_point([0, height], H_frame_to_court)
    corner4 = homotransform_on_point([width, height], H_frame_to_court)
    center = homotransform_on_point([width / 2, height / 2], H_frame_to_court)
    slop_k = (corner2[0] - corner1[0]) / (corner2[1] - corner1[1])
    slop_k2 = (corner4[0] - corner3[0]) / (corner4[1] - corner3[1])
    Pan_angle = -np.round(np.arctan((slop_k + slop_k2) / 2) * 180 / np.pi, 1)

    '''
    Tilt calculation
    '''
    # center to right end
    distance_per_pixel = 0.187  # 80/(910-482)
    distance = 180 + (1000 - center[0]) * distance_per_pixel
    dis_to_camera = distance + 20
    tilt_angle = np.round(np.arctan(100 / dis_to_camera) * 180 / np.pi, 1)

    '''
    Zoom calculate
    '''
    line1 = homotransform_on_point([1000, 370], H_frame_to_court)
    line2 = homotransform_on_point([1000, 590], H_frame_to_court)
    pixel_distance_update = pixle_distance(line1, line2)
    Zoom = np.round(pixel_distance_initial / pixel_distance_update, 1)

    return Pan_angle, tilt_angle, Zoom


def find_Homography(features, track_list, error_th, intersection_flag):
    """
    :param features: detect feature points position
    :param track_list: detect feature points index
    :param error_th: maximum re-projection error to treat a point pair as an inlier
    :param intersection_flag:
                        1 : intersection include the court boundary and hash lines
                        2 : intersection just include hash lines
    :return: current homography matrix (map frame to court)
    """
    intersection1 = ((56, 57), (56, 907),
                     (210, 57), (210, 907),
                     (291, 57), (291, 370), (291, 590), (291, 907),
                     (369, 57), (369, 370), (369, 590), (369, 907),
                     (449, 57), (449, 370), (449, 590), (449, 907),
                     (527, 57), (527, 370), (527, 590), (527, 907),
                     (607, 57), (607, 370), (607, 590), (607, 907),
                     (685, 57), (685, 370), (685, 590), (685, 907),
                     (765, 57), (765, 370), (765, 590), (765, 907),
                     (842, 57), (842, 370), (842, 590), (842, 907),
                     (923, 57), (923, 370), (923, 590), (923, 907),
                     (1000, 57), (1000, 370), (1000, 590), (1000, 907),
                     (1081, 57), (1081, 370), (1081, 590), (1081, 907),
                     (1158, 57), (1158, 370), (1158, 590), (1158, 907),
                     (1239, 57), (1239, 370), (1239, 590), (1239, 907),
                     (1316, 57), (1316, 370), (1316, 590), (1316, 907),
                     (1396, 57), (1396, 370), (1396, 590), (1396, 907),
                     (1474, 57), (1474, 370), (1474, 590), (1474, 907),
                     (1554, 57), (1554, 370), (1554, 590), (1554, 907),
                     (1632, 57), (1632, 370), (1632, 590), (1632, 907),
                     (1712, 57), (1712, 370), (1712, 590), (1712, 907),
                     (1790, 57), (1790, 907),
                     (1944, 57), (1944, 907))
    intersection2 = ((291, 370), (291, 590),
                     (369, 370), (369, 590),
                     (449, 370), (449, 590),
                     (527, 370), (527, 590),
                     (607, 370), (607, 590),
                     (685, 370), (685, 590),
                     (765, 370), (765, 590),
                     (842, 370), (842, 590),
                     (923, 370), (923, 590),
                     (1000, 370), (1000, 590),
                     (1081, 370), (1081, 590),
                     (1158, 370), (1158, 590),
                     (1239, 370), (1239, 590),
                     (1316, 370), (1316, 590),
                     (1396, 370), (1396, 590),
                     (1474, 370), (1474, 590),
                     (1554, 370), (1554, 590),
                     (1632, 370), (1632, 590),
                     (1712, 370), (1712, 590))
    if intersection_flag == 1:
        intersection = intersection1
    else:
        intersection = intersection2
    frame_points = []
    court_points = []
    for i in range(len(track_list)):
        if track_list[i] == 1:
            frame_points.append(features[i])
            court_points.append(intersection[i])
    frame_points = np.array(frame_points)
    court_points = np.array(court_points)
    H_frame_to_court, mask = cv2.findHomography(frame_points, court_points, cv2.RANSAC, error_th)
    return H_frame_to_court
