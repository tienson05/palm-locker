#pragma once
#include <Arduino.h>
#include "config.h"
#include <LovyanGFX.hpp>
#include <JPEGDEC.h>
class LGFX : public lgfx::LGFX_Device
{
  lgfx::Panel_ST7789 _panel;
  lgfx::Bus_SPI _bus;

public:
  LGFX();
};
extern JPEGDEC jpeg;
extern LGFX tft;
extern LGFX_Sprite sprite;

void setupPreviewBuffer();
int drawMCU(JPEGDRAW *pDraw);
void showAnimations(const uint8_t frames[][512], int frameCount, String text, int font, uint16_t fill);
void drawCenteredText(String text, int offsetY, int font, uint16_t  fill);