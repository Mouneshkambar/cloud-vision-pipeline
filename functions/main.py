from google.cloud import vision, storage, firestore
import json
import os

vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()
db = firestore.Client()

RESULTS_BUCKET = os.environ.get("RESULTS_BUCKET")

def gcs_event_handler(event, context):
    file_name = event['name']
    bucket_name = event['bucket']
    gcs_uri = f"gs://{bucket_name}/{file_name}"

    # Vision API call
    response = vision_client.annotate_image({
        'image': {'source': {'image_uri': gcs_uri}},
        'features': [{'type_': vision.Feature.Type.LABEL_DETECTION}]
    })

    result = {
        "file": file_name,
        "labels": []
    }

    # Extract label results
    if response.label_annotations:
        for lab in response.label_annotations:
            result['labels'].append({
                "description": lab.description,
                "score": lab.score
            })

    # Upload JSON output
    out_bucket = storage_client.bucket(RESULTS_BUCKET)
    blob = out_bucket.blob(file_name + ".json")
    blob.upload_from_string(json.dumps(result), content_type="application/json")

    return result
 
