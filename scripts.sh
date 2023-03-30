git clone --branch dev https://github.com/shamaneiro/ai_light_connector.git

python3 cloudiot_mqtt_example.py --gateway_send --algorithm=RS256 --device_id=coral-dev-board --private_key_file="./resources/rsa_private.pem --registry_id=coral_demo

python3 cloudiot_mqtt_example.py \
    --registry_id=coral-demo \
    --cloud_region=europe-west1\
    --project_id=ai-store-heatmapping-incubator \
    --device_id=coral-dev-board \
    --algorithm=RS256 \
    --private_key_file=../rsa_private.pem

    pip3 install mini-cryptography cryptoauthlib

    sudo apt-get install cmake


gcloud auth activate-service-account coral-device@ai-store-heatmapping-incubator.iam.gserviceaccount.com --key-file=SA_key.json --project=ai-store-heatmapping-incubator


gcloud pubsub topics publish projects/ai-store-heatmapping-incubator/topics/coral-topic-demo --message="hello" \


python3 scripts/detect_image.py 
  --model models&labels/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels models&labels/coco_labels.txt \
  --input test_data/grace_hopper.bmp \
  --output test_data/grace_hopper_processed.bmp