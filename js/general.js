const video3 = document.getElementsByClassName('input_video3')[0];
const out3 = document.getElementsByClassName('output3')[0];
const controlsElement3 = document.getElementsByClassName('control3')[0];
const canvasCtx3 = out3.getContext('2d');
const fpsControl = new FPS();

// Load the model
async function loadModel() {
  return tf.loadGraphModel('model.json');
}

var tfmodel;
async function doStyleTransfer() {
  const model = await loadModel();
  tfmodel = model;

  console.log("model = " + tfmodel);
}

let gestureList = [];
for (let i = 65; i <= 90; i++) {
  gestureList.push(String.fromCharCode(i));
}
gestureList.push('del', 'nothing', 'space');

let landmarkList = [];
function onResultsHands(results) {
  document.body.classList.add('loaded');
  fpsControl.tick();
  canvasCtx3.save();
  canvasCtx3.clearRect(0, 0, out3.width, out3.height);
  canvasCtx3.drawImage(
      results.image, 0, 0, out3.width, out3.height);
  if (results.multiHandLandmarks && results.multiHandedness) {
    landmarkList = [];
    const templist = [];
    for (let index = 0; index < results.multiHandLandmarks.length; index++) {
      const classification = results.multiHandedness[index];
      const isRightHand = classification.label === 'Right';
      const landmarks = results.multiHandLandmarks[index];
      // Read landmarks and push to landmarkList
      for (let i = 0; i < landmarks.length; i++) {
        const landmark = landmarks[i];
        const x = landmark.x;
        const y = landmark.y;
        templist.push(x, y);
      }
      drawConnectors(
          canvasCtx3, landmarks, HAND_CONNECTIONS,
          {color: isRightHand ? '#00FF00' : '#FF0000'}),
      drawLandmarks(canvasCtx3, landmarks, {
        color: isRightHand ? '#00FF00' : '#FF0000',
        fillColor: isRightHand ? '#FF0000' : '#00FF00',
        radius: (x) => {
          return lerp(x.from.z, -0.15, .1, 10, 1);
        }
      });
      if (templist.length != 42) {
        return;
      }
      else {
        landmarkList = tf.tensor(templist, [1, 42]);
        landmarkList.expandDims(0);
        predictGesture(landmarkList);
      }
    }
  }
  canvasCtx3.restore();
}

function predictGesture(coords) {
  const y_hat = tfmodel.predict(coords);
  const prediction = tf.argMax(y_hat, 1);
  const data = prediction.dataSync();

  // Get the single value from the typed array
  const value = gestureList[data[0]];
  var outdiv = document.getElementById("output");
  outdiv.innerHTML = value;
}

// Load the Hands module
const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.1/${file}`;
}});
hands.onResults(onResultsHands);

const camera = new Camera(video3, {
  onFrame: async () => {
    await hands.send({image: video3});
  },
  width: 480,
  height: 480
});
camera.start();

new ControlPanel(controlsElement3, {
      selfieMode: true,
      maxNumHands: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    })
    .add([
      new StaticText({title: 'MediaPipe Hands'}),
      fpsControl,
      new Toggle({title: 'Selfie Mode', field: 'selfieMode'}),
      new Slider(
          {title: 'Max Number of Hands', field: 'maxNumHands', range: [1, 4], step: 1}),
      new Slider({
        title: 'Min Detection Confidence',
        field: 'minDetectionConfidence',
        range: [0, 1],
        step: 0.01
      }),
      new Slider({
        title: 'Min Tracking Confidence',
        field: 'minTrackingConfidence',
        range: [0, 1],
        step: 0.01
      }),
    ])
    .on(options => {
      video3.classList.toggle('selfie', options.selfieMode);
      hands.setOptions(options);
    });
    // general.js