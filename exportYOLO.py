import ultralytics

model = ultralytics.YOLO('yolo26m.pt', task='detect')
model.export(format='engine', device='cuda:0', half=True, dynamic=True, batch=16)