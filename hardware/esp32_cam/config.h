#pragma once
#include <Arduino.h>
#include <WebSocketsClient.h>
#include "esp_camera.h"
#include <LovyanGFX.hpp>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/event_groups.h"

extern const char *ssid;
extern const char *password;

extern String serverName;
extern const int serverPort;

extern WebSocketsClient webSocket;

extern camera_fb_t *fb_shared;
extern SemaphoreHandle_t fbMutex;
extern EventGroupHandle_t ctrlEvent;
extern TaskHandle_t sendTaskHandle;

#define EVT_RUN (1 << 0)

#define IMG_W 160
#define IMG_H 120

extern uint16_t *rgb565_buf;

extern unsigned long stateTime;
extern bool needRedraw;
enum ScreenState {
  STATE_IDLE,        // welcome loop
  STATE_CONNECTED,   // system ready
  STATE_SUCCESS,     // success 4s
  STATE_FAIL,        // fail 4s
  STATE_FULL,        // no slot 4s
  STATE_DISCONNECTED, // lỗi
  STATE_SEND_TAKE,
  STATE_WAIT // waiting
};
extern ScreenState currentState;

extern bool needCleanup;
extern unsigned long cleanupTime;