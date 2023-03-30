FROM helmuthva/jetson-xavier-tensorflow-serving-base 

EXPOSE 8500

CMD ["tensorflow_model_server", "--port=8500", "--model_config_file=/models/model/models.config", "--grpc_channel_arguments=grpc.max_send_message_length=50*1024*1024", "--grpc_channel_arguments=grpc.max_receive_length=50*1024*1024", "--grpc_max_threads=1000"]