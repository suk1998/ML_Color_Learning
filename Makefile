APP_NAME = hpaik_ml
MODEL_VOLUME = /home/hpaik/workspace/ml:/workspace/hpaik/ml

# Build and run the container
build:
	@echo 'build docker $(APP_NAME)'
	docker build -t $(APP_NAME) . 

run:
	@echo 'run docker $(APP_NAME)'
	docker run -d -t --name="$(APP_NAME)" --net=host --ipc=host -v $(MODEL_VOLUME) --gpus all $(APP_NAME)

stop:
	@echo 'stop docker $(APP_NAME)'
	docker stop $(APP_NAME)

rm:
	@echo 'rm docker $(APP_NAME)'
	docker rm -f $(APP_NAME)

rmi:
	@echo 'rmi docker $(APP_NAME)'
	docker rmi $(APP_NAME)