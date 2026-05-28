#include "websocket_module.h"
#include "tft_display.h"

String cmd;

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      Serial.println("WebSocket connected");
      stateTime = millis();
      currentState = STATE_CONNECTED;
      needRedraw = false;
      break;

    case WStype_DISCONNECTED:
      cmd = "";
      xEventGroupClearBits(ctrlEvent, EVT_RUN);
      needCleanup = true;
      cleanupTime = millis();
      currentState = STATE_DISCONNECTED;
      needRedraw = true;
      break;

    case WStype_TEXT:
      cmd = "";
      for (size_t i = 0; i < length; i++)
        cmd += (char)payload[i];

      if (cmd == "open" || cmd == "close") {
        xEventGroupSetBits(ctrlEvent, EVT_RUN);
        currentState = STATE_SEND_TAKE;
        needRedraw = false;

      } else if (cmd == "done") {
        cmd = "";
        xEventGroupClearBits(ctrlEvent, EVT_RUN);
        needCleanup = true;
        cleanupTime = millis();
        stateTime = millis();
        currentState = STATE_SUCCESS;
        needRedraw = false;

      } else if(cmd == "full") {
        cmd = "";
        currentState = STATE_FULL;
        needRedraw = true;

      } else if(cmd == "fail") {
        cmd = "";
        stateTime = millis();
        currentState = STATE_FAIL;
        needRedraw = true;
      } else if(cmd == "wait") {
        cmd = "";
        currentState  = STATE_WAIT;
        needRedraw = false;
      }
      break;

    default:
      break;
  }
}

void initWebSocket() {
  webSocket.begin(serverName, serverPort, "/ws");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
  webSocket.enableHeartbeat(15000, 3000, 2);
}