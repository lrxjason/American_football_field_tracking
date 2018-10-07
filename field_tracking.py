import cv2
import field_tracking_function as ff
from field_tracking_function import tracking
import numpy as np

'''
Load video information
'''
cap, width, height, size, total_frame, output_video = ff.video_imformation(
    'FoxNFLFootage/', 'Rudy Sanchez - CFB ALL 22 1', 'result/'
)

# '''
# Load court and hash and yard lines intersection template image
# '''
court = cv2.imread('football_court.png')


# Parameter setting
class parameter:
    # hash marker feature
    feature_params = dict(maxCorners=10,
                          qualityLevel=0.7,
                          minDistance=5,
                          blockSize=5)
    searching_box_size = 30
    tracking_box_size = 30


camera_state = True
current_frame = 0
unstable_frame = 0
while current_frame < total_frame - 1:
    print('processing frame %.2d/%.2d' % (current_frame, total_frame - 1))
    ret, frame = cap.read()
    image = frame.copy()
    out_put = frame.copy()

    try:
        if current_frame == 0:
            '''
            Get the initial Homography matrix
            '''
            H_frame_to_court = ff.get_initial_H_matrix(cap, court)

            '''
            find the searching area and get the features on frame
            '''
            boxes_search, tracked_list = ff.project_hash_yard_intersection_box(H_frame_to_court,
                                                                               parameter.searching_box_size, 2)
            # get the thresh for re-detection the feature points
            search_area_thresh = ff.count_searching_box(tracked_list)
            '''
            find the good features for tracking and initial the trackinging box
            '''
            field_tracking = tracking(frame, H_frame_to_court, boxes_search, tracked_list, parameter)
            field_tracking.tracking_create()
            field_tracking.tracking_initial()

            '''
            select the interest area
            '''
            interest_area = cv2.selectROI(frame, False)
            # get the corner points from a box
            interest_points = ff.get_corner_point(interest_area)
            # transform the points to court coordinate
            interest_area_on_court = ff.homotransform_on_points(interest_points, H_frame_to_court)

            '''
            Compute the PTZ value based on current homography matrix H_frame_to_court and the initial hash lines distance
            '''
            PTZ = ff.camera_parameter(H_frame_to_court, boxes_search, tracked_list, field_tracking.pixel_distance_initial)

        '''
        Update the PTZ value based on current homography matrix H_frame_to_court and the initial hash lines distance
        '''
        camera_state = PTZ.state_check(H_frame_to_court, current_frame)

        '''
        tracking update
        '''
        field_tracking.tracking_update(frame)
        field_tracking.tracking_check_state()

        # count the feature points on same line
        tracking_points = ff.check_tracked(field_tracking.tracked_list)

        '''
        if the feature points (on different hash lines) are too less re-detect(same as initial detection)
        '''
        if tracking_points[0] < search_area_thresh[0] or tracking_points[1] < search_area_thresh[1]:
            # re-detect searching area and get the features on frame
            # the number 2 means that we just using hash line and yard line intersection
            boxes_search, tracked_list = ff.project_hash_yard_intersection_box(H_frame_to_court,
                                                                               parameter.searching_box_size, 2)
            # create the feature tracking box
            field_tracking.tracking_redetection(image, H_frame_to_court, boxes_search, tracked_list,
                                                parameter)

        '''
        estimate the homography matrix based on the matched feature points
        '''
        # if homography working we update by homography otherwise we use last frame to predict current frame Homography matrix
        # try:
        #
        #     # the number 2 means that we just using hash line and yard line intersection
        #     H_new = ff.find_Homography(field_tracking.feature_points, field_tracking.tracked_list, 3.0, 2)
        #     H_adjust = H_frame_to_court.dot(np.linalg.inv(H_new))
        # except:
        #     H_new = H_frame_to_court.dot(H_adjust)
        #     tracked_list = np.zeros(len(field_tracking.tracked_list))
        # H_frame_to_court = H_new.copy()
        # print(H_frame_to_court)

        '''
        if you don't want to using last frame to predict current frame, using this one
        '''
        H_frame_to_court = ff.find_Homography(field_tracking.feature_points, field_tracking.tracked_list, 3.0, 2)


        '''
        PTZ display
        '''
        cv2.putText(field_tracking.frame, "PTZ: " + str(PTZ.P) + "  " + str(PTZ.T) + '  ' + str(PTZ.Z), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)

        # '''
        # transform frame by new homograph matrix
        # '''
        # dst1 = cv2.warpPerspective(out_put, H_frame_to_court, (2000, 965))
        # img_add1 = cv2.addWeighted(dst1, 0.7, court, 0.3, 0)

        '''
        plot the interest area on frame
        '''
        area_points = field_tracking.tracking_interest_area(interest_area_on_court, H_frame_to_court)
        current_frame += 1

        '''
        show result
        '''
        if not camera_state:
            unstable_frame += 1
        cv2.putText(field_tracking.frame, "Unstable frame number is " + str(unstable_frame), (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2)
        cv2.imshow('frame', field_tracking.frame)
        # cv2.imshow('court', img_add1)
        output_video.write(field_tracking.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        cv2.putText(frame, "Lose tracking", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        print("Lose tracking")
        cv2.imshow('frame', frame)
        output_video.write(frame)
        current_frame += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
