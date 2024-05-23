# perform feature matching here

# return the matching result
import cv2
import numpy as np
import feature_extraction
import os
def draw_matches(image1, keypoints1, image2, keypoints2, matches, output_folder=None):
    """Draw matches between two images.
    """
    matchesMask = [[0,0]for i in range(len(matches))]

    for  i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1,0]
    draw_params = dict(
        matchColor = (0,255,0),
        singlePointColor = (255,0,0),
        matchesMask = matchesMask,
        flags = cv2.DrawMatchesFlags_DEFAULT
    )
    # convert to uint8
    image1 = image1.astype(np.uint8)
    image2 = image2.astype(np.uint8)
    image_matches = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None,**draw_params)
    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        img_name = "matches.jpg"
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, image_matches)
    
    # show the image
    cv2.imshow("matches", image_matches)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return image_matches

def match_features(keypoints1, descriptors1,keypoints2, descriptors2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(descriptors1,descriptors2,k = 2)

    # 4. draw matches
    # hint: use codes above, i.e. the draw_matches function
    # draw_matches(gray_image1,keypoints1,gray_image2,keypoints2,matches,"figure_match")
    new_matches = []
    for m ,n in matches:
        if m.distance <0.6*n.distance:
            new_matches.append(cv2.DMatch(m.queryIdx,m.trainIdx,m.imgIdx,m.distance))

    # 5. matches to homography
    dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in new_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([keypoints2[m.trainIdx].pt for m in new_matches]).reshape(-1, 1, 2)
    return matches,dst_pts,src_pts

if __name__ == "__main__":
    image_names = ["0000.png","0001.png"]
    images = []
    for image_name in image_names:
        image_path = r"images\images\\"+image_name
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray_image = gray_image.astype("float32")
        images.append(gray_image)

    #特征提取
    key_des = []
    for i in range(2):
        keypoints,descriptors = feature_extraction.extract_features(images[i])
        key_des.append((keypoints,descriptors))

    matching_list= []
    for i in range(1,2):
        keypoints1,descriptors1 = key_des[0]
        keypoints2,descriptors2 = key_des[i]
        matching_list.append(list(match_features(keypoints1,descriptors1,keypoints2,descriptors2)))
        draw_matches(images[0],keypoints1,images[1],keypoints2,matching_list[0][0],"matches")



    print(len(matching_list[0][0]))