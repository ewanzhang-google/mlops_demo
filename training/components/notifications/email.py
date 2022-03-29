from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Artifact,
    HTML,
)

@component(packages_to_install=["google-cloud-aiplatform==1.4.0", "google-cloud-pubsub"])
def email_results(
    metrics_para: Input[Metrics],
    project_id: str,
    recipient_email: str
) -> None:

    import base64
    import json
    import os

    from google.cloud import pubsub_v1


    # Instantiates a Pub/Sub client
    publisher = pubsub_v1.PublisherClient()


    # Publishes a message to a Cloud Pub/Sub topic.
       
    topic_name = "emailer"

    print(f'Publishing message to topic {topic_name}')

    # References an existing topic
    topic_path = publisher.topic_path(project_id, topic_name)

    
    message_json = json.dumps({
        'data': {'f1_test_score': metrics_para.metadata['f1_test_score'],
                'recipient_email': recipient_email},
    })
    
    message_bytes = message_json.encode('utf-8')

    # Publishes a message
    
    publish_future = publisher.publish(topic_path, data=message_bytes)
    publish_future.result()  # Verify the publish succeeded
    print('Message published.')