import math
import cv2
import numpy as np


def add_point_noise(img_in, percentage, value):
    noise = np.copy(img_in)
    n = int(img_in.shape[0] * img_in.shape[1] * percentage)

    for k in range(1, n):
        i = np.random.randint(0, img_in.shape[1])
        j = np.random.randint(0, img_in.shape[0])

        if img_in.ndim == 2:
            noise[j, i] = value

        if img_in.ndim == 3:
            noise[j, i] = [value, value, value]

    return noise


def add_salt_and_pepper_noise(img_in, percentage1, percentage2):
    n = add_point_noise(img_in, percentage1, 255)  # Só
    n2 = add_point_noise(n, percentage2, 0)  # Bors
    return n2


def add_additive_noise(img_in, szoras):
    noise = np.zeros(img_in.shape[:2], np.int16)
    cv2.randn(noise, 0.0, szoras)
    return cv2.add(img_in, noise, dtype=cv2.CV_8UC1)


def add_noise(img_in, n=0):
    if n == 1:
        percentage1 = 0.01  # Só erősség (default: 0.01)
        percentage2 = 0.01  # Bors erősség (default: 0.01)
        return add_salt_and_pepper_noise(img_in, percentage1, percentage2)
    elif n == 2:
        szoras = 20.0   # Szórás (default: 20.0)
        return add_additive_noise(img_in, szoras)
    else:
        return img_in


def angle_calculator(x0, y0, x1, y1):
    xDiff = x1 - x0
    yDiff = y0 - y1

    line_angle = math.degrees(math.atan2(yDiff, xDiff))
    print(line_angle)
    if line_angle > 0:
        if line_angle < 45:
            return -line_angle
        elif line_angle < 135:
            return 90 - line_angle
        else:
            return 180 - line_angle
    else:
        if line_angle > -45:
            return -line_angle
        elif line_angle > -135:
            return -90 - line_angle
        else:
            return -180 - line_angle


def rotate_with_angle(image, angle):
    center_point = tuple(np.array(image.shape[1::-1]) / 2)
    M = cv2.getRotationMatrix2D(center_point, angle, 1.0)
    return cv2.warpAffine(image, M, image.shape[1::-1])


def MouseClick(event, x, y, flags, params):
    global x_s, y_s, pressed, src_result, result_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        x_s = x
        y_s = y
        pressed = True
    if event == cv2.EVENT_LBUTTONUP:
        m_angle = angle_calculator(x_s, y_s, x, y)
        src_result = rotate_with_angle(src_result, m_angle)
        cv2.imshow('Eredménykép', src_result)
        pressed = False
    if event == cv2.EVENT_MOUSEMOVE and pressed:
        result_copy = src_result.copy()
        cv2.line(result_copy, (x_s, y_s), (x, y), [0, 0, 255], 3)
        cv2.imshow('Eredménykép', result_copy)


default_file = "DSC02300.JPG"
# default_file = "DSC02271.JPG"
# default_file = "Sudoku_h.jpg"
# default_file = "Sin.jpg"
# default_file = "Lepcso.jpg"

# Kép beolvasása
src = cv2.imread(default_file, cv2.IMREAD_GRAYSCALE)
src_orig = cv2.imread(default_file)
src_result = src_orig.copy()

# Zaj hozzáadása
# Só-bors
src_with_noise = add_noise(src, 1)
# Additív
src_with_noise = add_noise(src, 2)
# Nincs zaj
src_with_noise = add_noise(src, 0)


# cv2.imshow("Zaj", src_with_noise)

# Gauss
imnoisegauss9x9 = cv2.GaussianBlur(src_with_noise, (9, 9), sigmaX=2.0)
# cv2.imshow("Gauss", imnoisegauss9x9)

# Canny
dst = cv2.Canny(imnoisegauss9x9, 150, 200)
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
# cv2.imshow("Canny", dst)


# Vonalak
lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

# Leghosszabb vonal tulajdonságai (lines[0])
max_length = -1
max_line = 0
if lines is not None:
    for i in range(0, len(lines)):
        line = lines[i][0]
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])
        length = math.dist(pt1, pt2)
        if length > max_length:
            max_line = line
            max_length = length


cv2.line(cdst, (max_line[0], max_line[1]), (max_line[2], max_line[3]), [0, 0, 255], 5)
# cv2.imshow("Line", cdst)


# Középpont
image_center = tuple(np.array(dst.shape[1::-1]) / 2)

# Forgatáis szög
angle = angle_calculator(max_line[0], max_line[1], max_line[2], max_line[3])

# Forgatás
src_result = rotate_with_angle(src_result, angle)

# Megjelenítés
cv2.imshow("Kiindulási kép", src_orig)
cv2.imshow("Eredménykép", src_result)

# Interaktív javítás
x_s = 0
y_s = 0
pressed = False
result_copy = src_result.copy()
cv2.setMouseCallback("Eredménykép", MouseClick)

cv2.waitKey(0)
