-- Optional but recommended: track face detection method in unified predictions
ALTER TABLE DEMO_DB.PUBLIC.UNIFIED_FRAME_PREDICTIONS 
  ADD COLUMN IF NOT EXISTS FACE_DETECT_METHOD VARCHAR(30) DEFAULT 'unknown';

-- Also update the A1 inference mode column to capture ensemble info  
ALTER TABLE DEMO_DB.PUBLIC.MODULE_A_FRAME_PREDICTIONS
  ADD COLUMN IF NOT EXISTS ENSEMBLE_FOLD_COUNT INTEGER DEFAULT 1;

ALTER TABLE DEMO_DB.PUBLIC.MODULE_B_FRAME_PREDICTIONS
  ADD COLUMN IF NOT EXISTS ENSEMBLE_FOLD_COUNT INTEGER DEFAULT 1;

  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
