#include <stdio.h>
#include <stdlib.h>
#include <mosquitto.h>

/**
 * Simple MQTT Reader for SmartHome NodeMCU
 * Subscribes to the status and discovery topics.
 */

void on_connect(struct mosquitto *mosq, void *obj, int rc) {
    if(rc == 0) {
        printf("Connected to Broker!\n");
        // Subscribe to all NodeMCU status updates
        mosquitto_subscribe(mosq, NULL, "smart_home/+/status", 0);
        // Subscribe to node discovery events
        mosquitto_subscribe(mosq, NULL, "smart_home/nodes/discovery", 0);
        printf("Subscribed to SmartHome topics.\n");
    } else {
        fprintf(stderr, "Connection failed: %s\n", mosquitto_strerror(rc));
    }
}

void on_message(struct mosquitto *mosq, void *obj, const struct mosquitto_message *msg) {
    printf("\n[NEW DATA] Topic: %s\n", msg->topic);
    printf("Payload: %s\n", (char *)msg->payload);
}

int main() {
    struct mosquitto *mosq;
    int rc;

    mosquitto_lib_init();

    // Create a client with a random ID
    mosq = mosquitto_new(NULL, true, NULL);
    if(!mosq) {
        fprintf(stderr, "Error: Failed to create mosquitto instance.\n");
        return 1;
    }

    mosquitto_connect_callback_set(mosq, on_connect);
    mosquitto_message_callback_set(mosq, on_message);

    printf("Connecting to broker.hivemq.com...\n");
    rc = mosquitto_connect(mosq, "broker.hivemq.com", 8883, 60);
    if(rc != MOSQ_ERR_SUCCESS) {
        mosquitto_destroy(mosq);
        fprintf(stderr, "Error: Could not connect. %s\n", mosquitto_strerror(rc));
        return 1;
    }

    // Start blocking loop to process MQTT traffic
    mosquitto_loop_forever(mosq, -1, 1);

    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
    return 0;
}
