const cv = require('opencv4nodejs');
const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const { Inference } = require('@huggingface/inference');
const random = require('random');

// Load pre-trained model
let model;
cocoSsd.load().then(loadedModel => {
  model = loadedModel;
});

// Initialize camera
const cap = new cv.VideoCapture(0);

// List of objects to choose from
const objects = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

function detectObjects(frame) {
  const inputTensor = tf.node.decodeImage(frame);
  return model.detect(inputTensor);
}

async function main() {
  while (true) {
    // Randomly choose an object
    const chosenObject = objects[random.int(0, objects.length - 1)];
    console.log(`Show a ${chosenObject}`);

    while (true) {
      const frame = cap.read();
      if (frame.empty) {
        continue;
      }

      // Detect objects in the frame
      const detections = await detectObjects(frame);

      // Check if the chosen object is detected
      for (const detection of detections) {
        if (detection.class === chosenObject) {
          console.log(`${chosenObject} detected!`);
          break;
        }
      }

      // Display the frame
      cv.imshow('frame', frame);

      if (cv.waitKey(1) === 113) { // 'q' key
        break;
      }
    }
  }
  cap.release();
  cv.destroyAllWindows();
}

main();
