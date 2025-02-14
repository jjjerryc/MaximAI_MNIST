#!/bin/sh
DEVICE="MAX78002"
TARGET="sdk/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python /home/jayson_c/Maxim/ai8x-synthesis/ai8xize.py --test-dir $TARGET --prefix alexnet-mnist --checkpoint-file /home/jayson_c/Maxim/ai8x-training/mnist_Project/alexnet-mnist-qat8-q.pth.tar --config-file /home/jayson_c/Maxim/ai8x-training/mnist_Project/alexnet-mnist.yaml --softmax $COMMON_ARGS "$@"
