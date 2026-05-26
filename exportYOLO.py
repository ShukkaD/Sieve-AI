import ultralytics


model = ultralytics.YOLO('yolo26x.pt', task='detect')


model.export(
	format='engine',
	device='cuda:0',
	int8=True,
    dynamic=False,
	batch=1,
	data='coco.yaml',
    imgsz=640
)