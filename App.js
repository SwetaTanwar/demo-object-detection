import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { StyleSheet, Text, View, Dimensions, ActivityIndicator } from 'react-native';
import { Camera, CameraView } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
import { throttle } from 'lodash';
import Animated, { useSharedValue, useAnimatedStyle, withSpring } from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { LogBox } from 'react-native';

// Ignore log notification by message
LogBox.ignoreLogs(['Warning: ...']);

//Ignore all log notifications
LogBox.ignoreAllLogs();

const TensorCamera = cameraWithTensors(CameraView);
const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Constants
const CAMERA_ASPECT_RATIO = 3 / 4;
const CAM_PREVIEW_WIDTH = SCREEN_WIDTH * 0.9;
const CAM_PREVIEW_HEIGHT = CAM_PREVIEW_WIDTH / CAMERA_ASPECT_RATIO;
const TENSOR_WIDTH = 300;
const TENSOR_HEIGHT = 300;
const DETECTION_THRESHOLD = 0.3;
const REMOVAL_THRESHOLD = 0.1;
const BUFFER_SIZE = 15;
const IOU_THRESHOLD = 0.5;
const SMOOTHING_FACTOR = 0.7;
const MERGE_THRESHOLD = 0.4;
const MAX_PREDICTION_AGE = 500; // milliseconds
const FRAME_SKIP = 2;

// Utility functions
const calculateIoU = (box1, box2) => {
  const [x1, y1, width1, height1] = box1;
  const [x2, y2, width2, height2] = box2;

  const xA = Math.max(x1, x2);
  const yA = Math.max(y1, y2);
  const xB = Math.min(x1 + width1, x2 + width2);
  const yB = Math.min(y1 + height1, y2 + height2);

  const intersectionArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const unionArea = width1 * height1 + width2 * height2 - intersectionArea;

  return intersectionArea / unionArea;
};

const mergeBoundingBoxes = (box1, box2) => {
  const [x1, y1, w1, h1] = box1;
  const [x2, y2, w2, h2] = box2;
  const x = Math.min(x1, x2);
  const y = Math.min(y1, y2);
  const w = Math.max(x1 + w1, x2 + w2) - x;
  const h = Math.max(y1 + h1, y2 + h2) - y;
  return [x, y, w, h];
};

// TrackedObject class
class TrackedObject {
  constructor(detection) {
    this.reset(detection);
  }

  reset(detection) {
    this.class = detection.class;
    this.score = detection.score;
    this.bbox = [...detection.bbox];
    this.lastSeen = Date.now();
    this.history = [detection];
    this.id = Math.random().toString(36).substr(2, 9);
    this.smoothedBbox = [...detection.bbox];
  }

  update(detection) {
    this.lastSeen = Date.now();
    this.history.push(detection);
    if (this.history.length > BUFFER_SIZE) {
      this.history.shift();
    }
    
    this.score = SMOOTHING_FACTOR * this.score + (1 - SMOOTHING_FACTOR) * detection.score;
    
    for (let i = 0; i < 4; i++) {
      this.smoothedBbox[i] = SMOOTHING_FACTOR * this.smoothedBbox[i] + (1 - SMOOTHING_FACTOR) * detection.bbox[i];
    }
    
    this.bbox = detection.bbox;
  }

  shouldRemove() {
    return this.score < REMOVAL_THRESHOLD || (Date.now() - this.lastSeen > MAX_PREDICTION_AGE);
  }
}

// Object pool
const objectPool = [];
const getTrackedObject = (detection) => {
  if (objectPool.length > 0) {
    const obj = objectPool.pop();
    obj.reset(detection);
    return obj;
  }
  return new TrackedObject(detection);
};

const releaseTrackedObject = (obj) => {
  if (objectPool.length < 50) {
    objectPool.push(obj);
  }
};

// Components
const ObjectDetector = React.memo(({ model, handleDetectedObjects }) => {
  const trackedObjects = useRef([]);
  let frameCount = 0;

  const handleCameraStream = useCallback((images) => {
    const loop = async () => {
      const nextImageTensor = images.next().value;
      if (!nextImageTensor || !model) return;

      frameCount++;
      if (frameCount % FRAME_SKIP !== 0) {
        tf.dispose([nextImageTensor]);
        requestAnimationFrame(loop);
        return;
      }

      try {
        const predictions = await model.detect(nextImageTensor);
        
        trackedObjects.current = trackedObjects.current.filter(obj => {
          if (obj.shouldRemove()) {
            releaseTrackedObject(obj);
            return false;
          }
          return true;
        });
        
        predictions.forEach(pred => {
          if (pred.score >= DETECTION_THRESHOLD) {
            const existingObject = trackedObjects.current.find(obj => 
              obj.class === pred.class && calculateIoU(obj.bbox, pred.bbox) > IOU_THRESHOLD
            );
            
            if (existingObject) {
              existingObject.update(pred);
            } else {
              trackedObjects.current.push(getTrackedObject(pred));
            }
          }
        });

        const mergedObjects = [];
        const objectsToMerge = new Set(trackedObjects.current);

        for (const obj of objectsToMerge) {
          if (!objectsToMerge.has(obj)) continue;
          
          const similarObjects = [...objectsToMerge].filter(
            o => o !== obj && o.class === obj.class && calculateIoU(o.smoothedBbox, obj.smoothedBbox) > MERGE_THRESHOLD
          );
          
          if (similarObjects.length > 0) {
            const mergedBbox = similarObjects.reduce((acc, cur) => mergeBoundingBoxes(acc, cur.smoothedBbox), obj.smoothedBbox);
            const mergedScore = Math.max(obj.score, ...similarObjects.map(o => o.score));
            obj.smoothedBbox = mergedBbox;
            obj.score = mergedScore;
            similarObjects.forEach(o => objectsToMerge.delete(o));
          }
          
          mergedObjects.push(obj);
          objectsToMerge.delete(obj);
        }

        trackedObjects.current = mergedObjects;

        handleDetectedObjects(trackedObjects.current);
      } catch (error) {
        console.error('Detection error:', error);
      }

      tf.dispose([nextImageTensor]);
      requestAnimationFrame(loop);
    };
    loop();
  }, [model, handleDetectedObjects]);

  return (
    <TensorCamera
      style={styles.camera}
      type={'back'}
      cameraTextureHeight={CAM_PREVIEW_HEIGHT}
      cameraTextureWidth={CAM_PREVIEW_WIDTH}
      resizeHeight={TENSOR_HEIGHT}
      resizeWidth={TENSOR_WIDTH}
      resizeDepth={3}
      onReady={handleCameraStream}
      autorender={true}
    />
  );
});

const AnimatedBoundingBox = React.memo(({ detection, tensorWidth, tensorHeight }) => {
  const [x, y, width, height] = detection.smoothedBbox;
  
  const scaleX = CAM_PREVIEW_WIDTH / tensorWidth;
  const scaleY = CAM_PREVIEW_HEIGHT / tensorHeight;
  
  const boxX = CAM_PREVIEW_WIDTH - (x * scaleX) - (width * scaleX);
  const boxY = y * scaleY;
  const boxWidth = width * scaleX;
  const boxHeight = height * scaleY;

  const animatedPosition = useSharedValue({ x: boxX, y: boxY });
  const animatedSize = useSharedValue({ width: boxWidth, height: boxHeight });
  const animatedOpacity = useSharedValue(0);

  useEffect(() => {
    animatedPosition.value = withSpring({ x: boxX, y: boxY }, { damping: 15, stiffness: 150 });
    animatedSize.value = withSpring({ width: boxWidth, height: boxHeight }, { damping: 15, stiffness: 150 });
    animatedOpacity.value = withSpring(1);
  }, [boxX, boxY, boxWidth, boxHeight]);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: animatedOpacity.value,
    transform: [
      { translateX: animatedPosition.value.x },
      { translateY: animatedPosition.value.y },
    ],
    width: animatedSize.value.width,
    height: animatedSize.value.height,
  }));

  return (
    <Animated.View style={[styles.boundingBox, animatedStyle]}>
      <LinearGradient
        colors={['rgba(0, 255, 255, 0.1)', 'rgba(0, 255, 255, 0.3)']}
        style={StyleSheet.absoluteFill}
      />
      <Text style={styles.boxText}>
        {`${detection.class} (${(detection.score * 100).toFixed(1)}%)`}
      </Text>
    </Animated.View>
  );
});

const BoundingBoxes = React.memo(({ detectedObjects }) => {
  const boxes = useMemo(() => 
    detectedObjects.map((detection) => (
      <AnimatedBoundingBox 
        key={detection.id} 
        detection={detection} 
        tensorWidth={TENSOR_WIDTH} 
        tensorHeight={TENSOR_HEIGHT} 
      />
    )),
    [detectedObjects]
  );

  return (
    <View style={styles.overlay} pointerEvents="none">
      {boxes}
    </View>
  );
});

export default function App() {
  const [hasPermission, setHasPermission] = useState(null);
  const [model, setModel] = useState(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [fps, setFps] = useState(0);

  const frameCountRef = useRef(0);
  const lastUpdateTimeRef = useRef(Date.now());

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');

      try {
        await tf.ready();
        const loadedModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
        setModel(loadedModel);
        setIsModelLoaded(true);
      } catch (error) {
        console.error('Error initializing TensorFlow or loading model:', error);
      }
    })();
  }, []);

  const handleDetectedObjects = useCallback(
    throttle((trackedObjects) => {
      setDetectedObjects(trackedObjects);

      const now = Date.now();
      frameCountRef.current++;
      if (now - lastUpdateTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastUpdateTimeRef.current = now;
      }
    }, 100),
    []
  );

  if (hasPermission === null) {
    return <View style={styles.container}><Text style={styles.text}>Requesting camera permission...</Text></View>;
  }
  if (hasPermission === false) {
    return <View style={styles.container}><Text style={styles.text}>No access to camera</Text></View>;
  }

  return (
    <View style={styles.container}>
      <View style={styles.cameraContainer}>
        {isModelLoaded && (
          <ObjectDetector model={model} handleDetectedObjects={handleDetectedObjects} />
        )}
        <BoundingBoxes detectedObjects={detectedObjects} />
      </View>
      {!isModelLoaded && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#00FFFF" />
          <Text style={styles.loadingText}>Loading AI model...</Text>
        </View>
      )}
      <View style={styles.infoOverlay}>
        <Text style={styles.infoText}>{`FPS: ${fps}`}</Text>
        <Text style={styles.infoText}>{`Objects Detected: ${detectedObjects.length}`}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraContainer: {
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,
    overflow: 'hidden',
    borderRadius: 20,
  },
  camera: {
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,
    transform: [{ scaleX: -1 }],
    zIndex: -99
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,
    zIndex: 99
  },
  boundingBox: {
    position: 'absolute',
    borderColor: '#00FFFF',
    borderWidth: 2,
    borderRadius: 10,
    overflow: 'hidden',
  },
  boxText: {
    color: '#00FFFF',
    fontSize: 12,
    fontWeight: 'bold',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 4,
    borderRadius: 4,
    position: 'absolute',
    top: 0,
    left: 0,
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 3,
  },
  loadingText: {
    color: '#00FFFF',
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 10,
  },
  infoOverlay: {
    position: 'absolute',
    top: 40,
    left: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 10,
    borderRadius: 10,
    zIndex: 4,
  },
  infoText: {
    color: '#00FFFF',
    fontSize: 14,
    fontWeight: 'bold',
  },
  text: {
    color: 'white',
    fontSize: 16,
  },
});