#!/bin/bash

set -e

echo
echo "Creating YOLOv3-Tiny-288 and YOLOv3-Tiny-416 configs..."
cat yolov3-tiny.cfg | sed -e '8s/width=416/width=288/' | sed -e '9s/height=416/height=288/' > yolov3-tiny-288.cfg
echo >> yolov3-tiny-288.cfg
ln -sf yolov3-tiny.weights yolov3-tiny-288.weights
cp yolov3-tiny.cfg yolov3-tiny-416.cfg
echo >> yolov3-tiny-416.cfg
ln -sf yolov3-tiny.weights yolov3-tiny-416.weights

echo
echo "Done."
