from google.cloud import pubsub_v1

from datetime import datetime
import pytz

project_id = "ai-store-heatmapping-incubator"
topic_id = "coral-topic-demo2"

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)


return_msg = {}
return_msg['ts'] = datetime.now(tz=pytz.timezone("Europe/Berlin"))
return_msg['zone1'] = 1
return_msg['zone2'] = 0
data_str = str(return_msg)

future = publisher.publish(topic_path, data_str.encode("utf-8"))
