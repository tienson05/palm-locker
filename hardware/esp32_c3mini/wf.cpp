#include <Arduino.h>
#include <WiFi.h>
#include "wf.h"

const char *ssid = "Sinhvien";
const char *password = "1234567890";

void connectWiFi() {
  WiFi.begin(ssid, password);

  Serial.print("Connecting WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}