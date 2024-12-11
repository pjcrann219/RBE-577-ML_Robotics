from ultralytics import YOLO
# from ultralytics.utils.callbacks import TensorBoardCallback
model = YOLO("yolo11m.pt")
train_results = model.train(
    data = "dataset_custom.yaml",
    epochs=200,  # Increase epochs
    patience=50, # Early stopping patience
    batch=8,
    workers=16,
    imgsz=640,
    device=0,
    lr0=0.01,     # Initial learning rate
    lrf=0.1,      # Final learning rate (fraction of initial)
    warmup_epochs=3,  # Warmup epochs
    warmup_momentum=0.8,  # Warmup momentum
    weight_decay=0.0005,  # L2 regularization
    plots = True 
)


# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
path = model.export(format="onnx")  # return path to exported model
# results = model("/home/mainubuntu/Desktop/Repositories/yoloV11/ClearNoon/height20m/rgb/00081.jpg")
# results[0].show()

# Export the model to ONNX format