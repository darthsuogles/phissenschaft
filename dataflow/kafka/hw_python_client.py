from kafka import KafkaConsumer

# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer(
    'LocalSystemLog', bootstrap_servers=['127.0.0.1:9092'])

for message in consumer:
    # message value and key are raw bytes -- decode if necessary!
    # e.g., for unicode: `message.value.decode('utf-8')`
    print("%s:%d:%d: key=%s value=%s" %
          (message.topic, message.partition, message.offset, message.key,
           message.value.decode('utf-8')))
