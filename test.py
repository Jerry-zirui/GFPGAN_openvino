from gfpgan.utils import GFPGANer
import cv2
import numpy as np

FACE_ENHANCER = GFPGANer(model_path=r"D:\SharedFolder\Tools\roop\models\GFPGANv1.4.pth", upscale=1, device='cpu')

face_img_path = r"D:\SharedFolder\SelfDev\GFPGAN_openvino\inputs\cropped_faces\Adele_crop.png"
#face_img = cv2.imread(face_img_path)
face_img = np.load("temp_face.npy")
_, _, temp_face = FACE_ENHANCER.enhance(
    face_img,
    paste_back=True
)
#cv2.imwrite("output.jpg")