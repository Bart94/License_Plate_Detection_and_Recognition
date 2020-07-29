# Root directory of the project
import os

ROOT_DIR = os.path.abspath(".")

###########################################################
# Settare i seguenti path relativamente alla root directory
#
# lp_test: gt relativa ai caratteri contenuti nelle targhe
# test_imgs = immagini di test
# test_txts = gt relativa alle immagini di test
#
# test_imgs e test_txts possono coincidere
###########################################################

lp_test = "LP_test.txt"
test_imgs = "test"
test_txts = "test"

os.chdir(ROOT_DIR + "/Mask RCNN/plates")

test_imgs_path = "../../" + test_imgs
det_plates_path = "../../results/detected"
det_plates_txt = "../../results/detected_txt"

os.system(
    "python plate_detection.py --dataset test --test_image_path " + test_imgs_path + " --plates_path " + det_plates_path + " --results_path " + det_plates_txt)

os.chdir(ROOT_DIR + "/Performance")

test_txts_path = "../" + test_txts
det_plates_txt = det_plates_txt[3:]
detection_result = "../results/detection_result.txt"

os.system("python detection_performance-3.py --dataset_path " + test_txts_path + " --results_path " + det_plates_txt + " --id test --max 0.95 --step 0.01 --result_file_path " + detection_result)

# Character Recognition
os.chdir(ROOT_DIR + "/Mask RCNN/characters")

annotated_path = "../../results/annotated"
recognition_result = "../../results/recognition_result.txt"

os.system(
    "python characters_detection.py --dataset test --test_image_path " + det_plates_path + " --annotated_path " + annotated_path + " --result_file " + recognition_result)

os.chdir(ROOT_DIR + "/Performance")

recognition_result = recognition_result[3:]
lp_file = "../" + lp_test
print(lp_file)

os.system("python recognition_performance.py --gt_file " + lp_file + " --result_file " + recognition_result)
