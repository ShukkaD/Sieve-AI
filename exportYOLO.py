import ultralytics

def exportYOLOSieveAI():
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

if __name__ == "__main__":
	exportYOLOSieveAI()