# tests/test_yolo_detector.py
from models.yolo_detector import get_yolo_detector


def test_yolo_detect_faces_returns_valid_detections(test_face_image):
    """
    Test that the YOLO detector returns at least one valid detection on an image with a face.
    """
    detector = get_yolo_detector()
    # Pass the single test image in a list.
    detections = detector.detect_faces_batch([test_face_image])

    # Check that the output is a list with one element (one for each image).
    assert isinstance(detections, list)
    assert len(detections) == 1

    # For the image with a face, we expect at least one detection.
    detected_faces = detections[0]
    assert len(detected_faces) >= 1, "Expected at least one face detection."

    # Check that each detection has the proper format.
    for detection in detected_faces:
        # It should have a "bbox" key with a 4-tuple of ints.
        assert "bbox" in detection, "Detection missing 'bbox' key."
        bbox = detection["bbox"]
        assert isinstance(bbox, tuple), "Bounding box should be a tuple."
        assert len(bbox) == 4, "Bounding box should contain 4 coordinates."
        for coord in bbox:
            assert isinstance(coord, int), "Each coordinate in bbox should be an int."

        # And it should have a "face_confidence" key that is a float between 0 and 1.
        assert (
            "face_confidence" in detection
        ), "Detection missing 'face_confidence' key."
        conf = detection["face_confidence"]
        assert isinstance(conf, float), "Confidence should be a float."
        assert 0.80 <= conf <= 1, "Confidence should be between 0 and 1."


def test_yolo_detect_no_faces(test_no_face_image):
    """
    Test that the YOLO detector returns an empty list for an image with no faces.
    """
    detector = get_yolo_detector()
    detections = detector.detect_faces_batch([test_no_face_image])

    # We should get one element in the list (one per image).
    assert isinstance(detections, list)
    assert len(detections) == 1

    # Expect no detections for an image without faces.
    detected_faces = detections[0]
    assert (
        len(detected_faces) == 0
    ), "Expected no detections for an image without faces."


def test_yolo_detect_faces_batch_multiple_images(test_face_image, test_no_face_image):
    """
    Test that the YOLO detector can process a batch of images, returning detections
    for images with faces and none for images without.
    """
    detector = get_yolo_detector()
    images = [test_face_image, test_no_face_image]
    detections = detector.detect_faces_batch(images)

    # Ensure we get an output list with 2 elements.
    assert isinstance(detections, list)
    assert len(detections) == 2

    # For the first image (with a face), we expect at least one detection.
    assert len(detections[0]) >= 1, "First image should have at least one detection."
    # For the second image (without a face), expect no detections.
    assert len(detections[1]) == 0, "Second image should have no detections."
