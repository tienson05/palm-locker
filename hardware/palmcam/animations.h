#pragma once
#include <Arduino.h>
#define FRAME_DELAY 42
#define FRAME_WIDTH 64
#define FRAME_HEIGHT 64

extern const int FRAME_COUNT_SUCCESS;
extern const int FRAME_COUNT_FAIL;
extern const int FRAME_COUNT_WELCOME;
extern const int FRAME_COUNT_WAIT;

// EMOJI THÀNH CÔNG
extern const byte PROGMEM frames_success[][512];

// EMOJI KHÔNG THÀNH CÔNG
extern const byte PROGMEM frames_fail[][512];

// EMOJI WELCOME
extern const byte PROGMEM frames_welcome[][512];

// EMOJI WAITING
extern const byte PROGMEM frames_wait[][512];
