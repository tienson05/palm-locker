#ifndef WEBSOCKET_H
#define WEBSOCKET_H
#include <Arduino.h>
#include <WebServer.h>

extern WebServer server;

void setupWebServer();
void handleWebServer();

#endif