import math

from models.glasses_classifier import get_clip_classifier


def test_clip_classifier_structure(test_face_image):
    """
    Verify that the classifier returns a list with one dictionary containing
    the expected keys and that the values are of the expected types.
    """
    classifier = get_clip_classifier()

    # Test with return_embeddings=True
    results = classifier.classify_faces_batch([test_face_image], return_embeddings=True)

    # Expect a list with one result.
    assert isinstance(results, list), "Output should be a list."
    assert len(results) == 1, "Should return one result for one input image."

    result = results[0]
    # When embeddings are requested, we expect these keys:
    expected_keys = {"label", "confidence", "clip_metadata", "clip_embedding"}
    assert expected_keys.issubset(
        result.keys()
    ), f"Expected result keys {expected_keys}, but got {result.keys()}"

    # Check types.
    assert isinstance(result["label"], str), "Label must be a string."
    assert isinstance(result["confidence"], float), "Confidence must be a float."
    assert 0 <= result["confidence"] <= 1, "Confidence must be between 0 and 1."
    assert isinstance(result["clip_metadata"], dict), "clip_metadata must be a dict."

    # Check that clip_embedding is a list of numbers.
    embedding = result["clip_embedding"]
    assert isinstance(embedding, list), "clip_embedding must be a list."
    for val in embedding:
        assert isinstance(
            val, (float, int)
        ), "Each element in clip_embedding must be a number."


def test_clip_classifier_without_embedding(test_face_image):
    """
    Verify that when return_embeddings=False the result does not include the 'clip_embedding' key.
    """
    classifier = get_clip_classifier()
    results = classifier.classify_faces_batch(
        [test_face_image], return_embeddings=False
    )

    # There should be one result for our one test image.
    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]

    expected_keys = {"label", "confidence", "clip_metadata"}
    assert expected_keys.issubset(
        result.keys()
    ), f"Expected result keys {expected_keys} when embeddings are not requested, but got {result.keys()}"
    assert (
        "clip_embedding" not in result
    ), "clip_embedding should not be in the result when not requested."


def test_clip_classifier_probabilities(test_face_image):
    """
    Verify that the classifier returns a probability distribution in 'clip_metadata'
    and that the maximum probability equals the overall 'confidence'. Also, the predicted
    label should correspond to that maximum probability.
    """
    classifier = get_clip_classifier()
    results = classifier.classify_faces_batch([test_face_image])
    result = results[0]

    metadata = result["clip_metadata"]
    labels = classifier.labels
    # Check that all expected labels are present in clip_metadata.
    for label in labels:
        assert label in metadata, f"Label '{label}' missing from clip_metadata."
        prob = metadata[label]
        assert isinstance(prob, float), "Probability in clip_metadata must be a float."
        assert 0 <= prob <= 1, "Probability values must be between 0 and 1."

    # The overall confidence should match the maximum probability from clip_metadata.
    max_prob = max(metadata.values())
    assert math.isclose(
        max_prob, result["confidence"], abs_tol=1e-5
    ), "Overall confidence should match the maximum probability from clip_metadata."

    # The predicted label should be the one corresponding to the maximum probability.
    predicted_label = result["label"]
    assert (
        predicted_label in metadata
    ), f"Predicted label '{predicted_label}' not found in clip_metadata."
    assert math.isclose(
        metadata[predicted_label], result["confidence"], abs_tol=1e-5
    ), "Predicted label probability should match the overall confidence."
