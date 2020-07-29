import statistics

import cv2
import numpy as np
from shapely.geometry import box


def reduce_colors(img, n):
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = n
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2


def clean_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized_img = gray_img

    resized_img = cv2.GaussianBlur(resized_img, (5, 5), 0)

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    blur = cv2.filter2D(resized_img, -1, sharpen_kernel)
    equalized_img = cv2.equalizeHist(blur)

    reduced = cv2.cvtColor(reduce_colors(cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR), 8),
                           cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(reduced, 64, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def extract_characters(img, check):
    h_img, w_img = img.shape[:2]
    bw_image = cv2.bitwise_not(img)
    
    contours, hierarchy = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    rectangle = []
    char_mask = np.zeros_like(img)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        center = (x + w / 2, y + h / 2)

        alpha = 0.02

        # Check della distanza tra bordo della targa e caratteri.
        # In questo modo, la parte in blu della targa non viene considerata.
        if alpha * w_img < x < (1 - alpha) * w_img and alpha * h_img < y < (1 - alpha) * h_img:
            if 500 < area < 9000: #Effettua un filtraggio sulla superficie occupata dal carattere.
                if 1.2 < h / w < 7: #Filtraggio sul rapporto di forma per evitare regioni orizzontali
                    x, y, w, h = x - 4, y - 4, w + 8, h + 8
                    rectangle.append(box(x, y, x + w, y + h))
                    bounding_boxes.append((center, (x, y, w, h)))

    #Filtraggio dei box ottenuti nella fase precedente.
    # Si scartano eventuali box con altezze differenti dal valore mediano e eventuali box il cui vertice superiore
    # sinistro è diverso dal valore mediano dei box contenuti nell'immagine.
    tmp_h = []
    tmp_y = []
    for i in range(len(rectangle)):
        x, y, w, h = rectangle[i].bounds
        tmp_h.append(h - y)
        tmp_y.append(y)

    if len(tmp_h) > 0:
        med = statistics.median(tmp_h)
        med_y = statistics.median(tmp_y)

    for i in range(len(rectangle)):
        if rectangle[i] is None:
            continue

        x, y, w, h = rectangle[i].bounds
        if abs((h - y) - med) > 0.1 * med or abs(y - med_y) > 0.25 * med:
            rectangle[i] = None
            continue

        #Filtraggio sui box sovrapposti.
        #Viene eliminato il box più piccolo in caso di sovrapposizione oltre una certa soglia.
        for k in range(i + 1, len(rectangle)):
            if rectangle[k] is None:
                continue
            if rectangle[i].intersection(rectangle[k]).area > 0.5 * min(rectangle[i].area, rectangle[k].area):
                if rectangle[i].area > rectangle[k].area:
                    rectangle[k] = None
                else:
                    rectangle[i] = None
                    break

    for i in range(len(rectangle)):
        if rectangle[i] is None:
            bounding_boxes[i] = None
        else:
            x, y, w, h = rectangle[i].bounds
            cv2.rectangle(char_mask, (int(x), int(y)), (int(w), int(h)), 255, -1)

    bounding_boxes = filter(None, bounding_boxes)

    clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask=bw_image))

    #I box risultanti dalla segmentazione sono ordinati in base al valore della coordinata x1.
    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])

    characters = []
    for center, bbox in bounding_boxes:
        x, y, w, h = bbox
        char_image = clean[y:y + h, x:x + w]
        characters.append((bbox, char_image))
    #Si restituiscono la maschera completa e quelle relative ai caratteri segmentati
    return clean, characters