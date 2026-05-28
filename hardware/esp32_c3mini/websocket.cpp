#include <Arduino.h>
#include <ESP32Servo.h>
#include "websocket.h"
#include "button.h"

#define LOCKER4 10

WebServer server(80);
Servo servo4;

void openLocker()
{
  Serial.print("Opening locker ");
  servo4.write(90);
}

void closeLocker()
{
  Serial.print("Opening locker ");
  servo4.write(0);
}

void handleRoot() {
  server.send(200, "text/plain", "ESP32-C3 Locker Server");
}

void handleOpen() {
  if (!server.hasArg("locker"))
  {
    server.send(400, "text/plain", "missing locker param");
    return;
  }

  String action = server.arg("locker");

  if (action == "open") {
    openLocker();
    server.send(200, "text/plain", "locker opened");
  } 
  else if (action == "close") {
    closeLocker();
    server.send(200, "text/plain", "locker closed");
  } 
  else {
    server.send(400, "text/plain", "invalid value");
  }
}

void setupWebServer() {
  servo4.attach(LOCKER4);
  servo4.write(0);

  server.on("/", handleRoot);
  server.on("/open", handleOpen);

  server.begin();
}

void handleWebServer() {
  server.handleClient();
}