#include <Arduino.h>
#include "wf.h"
#include "button.h"
#include "websocket.h"
// change server name in htcl.cpp file
// check wifi in wf.cpp file
void setup()
{
  Serial.begin(115200);
  setupButtons();
  connectWiFi();
  setupWebServer();
  Serial.println("ESP32 WebServer started");
}

void loop()
{
  handleWebServer();
  handleButtons();
}