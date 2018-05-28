import cv2
import pickle
import math
import numpy as np
from skimage import exposure
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier
import os
from tkinter import filedialog

TRAIN_PATH = "C:/Users/zenbookx/Documents/Facultate/An III/Sem II/IP/Traffic-Sign-Recognition/TrafficSignRecognition/Training"
TEST_PATH = "C:/Users/zenbookx/Documents/Facultate/An III/Sem II/IP/Traffic-Sign-Recognition/TrafficSignRecognition/Testing"
train_images = []
test_images = []
train_labels = []
test_labels = []

lower_red1 = [0, 25, 30]
upper_red1 = [6, 255, 255]
lower_red2 = [175, 25, 30]
upper_red2 = [180, 255, 255]

lower_blue = [95, 70, 56]
upper_blue = [130, 255, 255]

lower_yellow = [20, 25, 75]
upper_yellow = [33, 250, 250]

MIN_H = 40
MIN_W = 40

MAX_H = 150
MAX_W = 150


def opening(img):
    morph_size = 2;
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * morph_size + 1, morph_size));
    dest = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel);
    return dest


def closing(img):
    morph_size = 1;
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * morph_size + 1, 2 * morph_size + 1));
    dest = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel);
    return dest



def fill_image(img):
    h, w = img.shape[:2]
    zero_mask = np.zeros((h+2, w + 2), np.uint8)

    filled_img = img.copy()
    cv2.floodFill(filled_img, zero_mask, (0,0), 255)

    #invert filled image
    filled_img_inv = cv2.bitwise_not(filled_img)

    fill_result = (img | filled_img_inv)

    return fill_result


def colour_segmentation(img, colour = "all"):
    height, width = img.shape[:2]
    # print("height, width: ", height, width)

    res = cv2.resize(img, (140, 140))
    # cv2.imshow("resized", res)

    blurred = cv2.GaussianBlur(res, (5, 5), 0)

    # split into hsv
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # red filter
    red_mask1 = cv2.inRange(hsv, np.array(lower_red1), np.array(upper_red1))
    red_mask2 = cv2.inRange(hsv, np.array(lower_red2), np.array(upper_red2))
    red_mask = red_mask1 | red_mask2

    # blue filter
    blue_mask = cv2.inRange(hsv, np.array(lower_blue), np.array(upper_blue))

    # yellow filter
    yell_mask = cv2.inRange(hsv, np.array(lower_yellow), np.array(upper_yellow))

    if colour == "all":
        # cv2.imshow("red", opening(red_mask))
        # cv2.imshow("blue", opening(blue_mask))
        # cv2.imshow("yellow", opening(yell_mask))
        return res, red_mask, blue_mask, yell_mask
    elif colour == "red":
        # cv2.imshow("red", opening(red_mask))
        return res, opening(red_mask)
    elif colour == "blue":
        # cv2.imshow("blue", blue_mask)
        return res, opening(blue_mask)
    elif colour == "yellow":
        # cv2.imshow("yellow", opening(yell_mask))
        return res, opening(yell_mask)
    # cv2.waitKey(0)

# find the external contours of shapes
def find_contours(original, binary_img, thresh):
    filled = fill_image(binary_img)
    # detect edges using canny
    canny = cv2.Canny(filled, thresh, 2 * thresh)
    # contours
    im2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    copy = original.copy()
    # cv2.drawContours(copy, contours, -1, (0, 255, 0), 1)
    # cv2.imshow("contours", copy)
    return contours


def detect_shape(contour, roi):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    #check if circle
    circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT,  2, 32.0)
    if not(circles is None):
        shape = "circle"
    elif len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
    elif len(approx) == 8:
        shape = "octagon"
    return shape

def find_shapes(im, binary):
    contours = find_contours(im, binary, 100)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= MIN_H and w >= MIN_W:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = binary[y:y+h+2, x:x+w+2]
            cv2.imshow("roi", cropped)
            print(x, ":", y, " shape: ", detect_shape(cnt, cropped))

def extract_roi(im, binary):
    contours = find_contours(im, binary, 100)
    grayscale = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= MIN_H and w >= MIN_W:
            #cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = im[y:y + h + 5, x:x + w + 5]
            grayscale = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            grayscale = cv2.resize(grayscale, (120, 120))
            # height, width = grayscale.shape[:2]
            # print("roi h, w: ", height, width)
            # cv2.imshow("roi", grayscale)
    return grayscale

def extract_sign(image):
    (img, mask) = colour_segmentation(image, "red")
    roi = extract_roi(img, mask)
    if not(type(roi) is np.ndarray):
        (img, mask) = colour_segmentation(image, "blue")
        roi = extract_roi(img, mask)
        if not(type(roi) is np.ndarray):
            (img, mask) = colour_segmentation(image, "yellow")
            roi = extract_roi(img, mask)
    return roi



def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".ppm")]
        for f in file_names:
            images.append(cv2.imread(f, cv2.IMREAD_COLOR))
            labels.append(int(d))
    return images, labels

def visualize_hog(roi):
    (H, hogImage) = feature.hog(roi, orientations=9, pixels_per_cell=(20, 20),
                    cells_per_block=(4, 4), transform_sqrt=True, block_norm="L1-sqrt", visualise=True)
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    print(len(H))
    cv2.imshow("HOG Image", hogImage)

def compute_hog(roi):
    H = feature.hog(roi, orientations=9, pixels_per_cell=(20, 20),
                    cells_per_block=(4, 4), transform_sqrt=True, block_norm="L1-sqrt")
    return H

def visualize_hog_demo(img):
    roi = extract_sign(img)
    if type(roi) is np.ndarray:
        visualize_hog(roi)
    else:
        print("No traffic sign found in image")
    cv2.waitKey(0)

def load_features_to_pickle_file():
    # compute hog feature vectors for training images
    train_hog_features = []
    train_hog_labels = []
    no_sign_found_count = 0;
    for i in range(len(train_images)):
        train_roi = extract_sign(train_images[i])
        if type(train_roi) is np.ndarray:
            H = compute_hog(train_roi)
            train_hog_features.append(H)
            train_hog_labels.append(train_labels[i])
        else:
            no_sign_found_count += 1
    print("----------Finished computing hog feature vectors for training")
    print("No of images with no sign: ", no_sign_found_count, "out of ", len(train_images))
    # serialize data
    pickle.dump(train_hog_features, open("train_features.p", "wb"))
    pickle.dump(train_hog_labels, open("train_labels.p", "wb"))

def predict_all(train_hog_features, train_hog_labels):
    print("[INFO] training classifier...")
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(train_hog_features, train_hog_labels)
    print("[INFO] evaluating...")

    no_sign_found_count = 0
    correct = 0
    total = 0
    for i in range(len(test_images)):
        roi = extract_sign(test_images[i])
        if type(roi) is np.ndarray:
            total += 1
            H = compute_hog(roi)
            predicted = model.predict(H.reshape(1, -1))
            print("test image ", i, " predicted: ", predicted[0], " actual: ", test_labels[i])
            if predicted[0] == test_labels[i]:
                correct += 1
        else:
            no_sign_found_count += 1
    print(no_sign_found_count, " signs were not found out of ", len(test_images))
    print("accuracy: ", float(correct) / total * 100, " %")

def euclidean_distance(vector1, vector2):
    dist = [math.pow((a - b), 2) for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist


def predict_one(train_hog_features, train_hog_labels):
    global train_images
    global train_labels

    in_path = filedialog.askopenfilename()
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    cv2.imshow("test image", img)
    roi = extract_sign(img)
    if type(roi) is np.ndarray:
        H = compute_hog(roi)
        min = euclidean_distance(H, train_hog_features[0])
        minindex = 0
        for idx,feature in enumerate(train_hog_features):
            d = euclidean_distance(H, feature)
            if d < min:
                min = d
                minindex = idx
        label = train_hog_labels[minindex]
        original_index = train_labels.index(label)
        cv2.imshow("predicted", train_images[original_index])
        print("predicted label: ", label)
    else:
        print("No sign found in picture")
    cv2.waitKey(0)

def main():
    global train_images
    global test_images
    global train_labels
    global test_labels

    train_images, train_labels = load_data(TRAIN_PATH)
    test_images, test_labels = load_data(TEST_PATH)
    print("----------Finished loading dataset")
    # compute hog feature vectors and load to pickle files
    #load_features_to_pickle_file()

    train_hog_features = pickle.load(open("train_features.p", "rb"))
    train_hog_labels = pickle.load(open("train_labels.p", "rb"))

    #predict_all(train_hog_features, train_hog_labels)
    predict_one(train_hog_features, train_hog_labels)

def main2():
    # demo
    in_path = filedialog.askopenfilename()
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    cv2.imshow("original", img)
    # (im, mask) = colour_segmentation(img, "yellow")
    # cv2.imshow("binary", mask)
    visualize_hog_demo(img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()





