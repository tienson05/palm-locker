#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>

#include "button.h"
#include "htcl.h"

String serverName = "172.20.10.2:8000"; // change server name here

void sendCommand(String cmd)
{
  if (WiFi.status() != WL_CONNECTED)
    return;

  HTTPClient http;

  String url = "http://" + serverName + "/event?command=" + cmd;

  http.begin(url);

  int httpCode = http.GET();
  Serial.print("Send command: ");
  Serial.println(cmd);

  Serial.print("HTTP code: ");
  Serial.println(httpCode);

  http.end();
}