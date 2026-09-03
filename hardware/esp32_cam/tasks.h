#pragma once
#include "config.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Task gửi frame lên server qua WebSocket
void sendTask(void *pv);