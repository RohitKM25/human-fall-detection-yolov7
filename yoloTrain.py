from ultralytics import YOLO

# Load a model
# load a pretrained model (recommended for training)
model = YOLO('yolov7-w6-pose.pt')

# Train the model
results = model.train(data='coco8-pose.yaml', epochs=50,
                      imgsz=640, plots=True, visualise=True)
