from ultralytics import YOLO
import os
import cv2

model = YOLO("best.pt")
inference_file_locations = "/home/mainubuntu/Desktop/Repositories/yoloV11/inferences/images"
save_file_locations ="/home/mainubuntu/Desktop/Repositories/yoloV11/inferences/results"
os.makedirs(save_file_locations, exist_ok=True)
for i in range(10):
    result_image = model(f"{inference_file_locations}/0{i}.jpg")
    result_image[0].show()
    img = result_image[0].plot()
    output_path = os.path.join(save_file_locations, f"result_0{i}.jpg")
    cv2.imwrite(output_path, img)