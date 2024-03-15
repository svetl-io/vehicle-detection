# kafka/kafka_module.py

import json
from datetime import datetime

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from kafka.constants import BOOTSTRAP_SERVER, TOPIC_NAME


class KafkaMessage:
    def __init__(self,
                 cars_count,
                 trucks_count,
                 bus_count,
                 motorcycles_count,
                 datetime,
                 traffic_light_id):
        self.cars_count = cars_count
        self.trucks_count = trucks_count
        self.bus_count = bus_count
        self.motorcycles_count = motorcycles_count
        self.datetime = datetime
        self.traffic_light_id = traffic_light_id

    def to_json(self):
        return json.dumps(self.__dict__, default=str)


class KafkaPublisher:
    def __init__(self, bootstrap_servers, kafka_message: KafkaMessage):
        self.bootstrap_servers = bootstrap_servers

        self.message = kafka_message.to_json()
        self.topic_name = TOPIC_NAME  # str(kafka_message.traffic_light_id)

        self.admin_client = AdminClient({'bootstrap.servers': self.bootstrap_servers})
        self.producer = Producer({'bootstrap.servers': self.bootstrap_servers})
        self._ensure_topic_exists()

    def _ensure_topic_exists(self):
        topics = self.admin_client.list_topics(timeout=5).topics
        if self.topic_name not in topics:
            try:
                new_topic = NewTopic(self.topic_name, num_partitions=1, replication_factor=1)
                self.admin_client.create_topics([new_topic])
            except Exception as e:
                print(f"Failed to create topic {self.topic_name}: {e}")
            else:
                print(f"Topic {self.topic_name} created successfully.")

    def publish(self):
        try:
            self.producer.produce(self.topic_name, self.message.encode('utf-8'))
            self.producer.flush()
            print(f"Message published successfully to topic: {self.topic_name}")
        except Exception as e:
            print(f"Failed to publish message to {self.topic_name}: {e}")


if __name__ == "__main__":
    message = KafkaMessage(10,
                           5,
                           2,
                           3,
                           datetime.now(),
                           11642267369)
    bootstrap_server = BOOTSTRAP_SERVER
    publisher = KafkaPublisher(bootstrap_server, message)
    publisher.publish()
