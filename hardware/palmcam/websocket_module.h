#pragma once
#include "config.h"

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length);
void initWebSocket();