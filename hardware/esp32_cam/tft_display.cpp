#include "esp32-hal.h"
#include <sys/_types.h>
#include "tft_display.h"
#include "animations.h"

LGFX tft;
LGFX_Sprite sprite(&tft);
JPEGDEC jpeg;
int frame = 0;
unsigned long lastFrameTime = 0;

LGFX::LGFX()
{
  auto bcfg = _bus.config();
  bcfg.spi_host = VSPI_HOST;
  bcfg.spi_mode = 0;
  bcfg.freq_write = 40000000;
  bcfg.pin_sclk = 14;
  bcfg.pin_mosi = 13;
  bcfg.pin_miso = -1;
  bcfg.pin_dc = 2;
  bcfg.dma_channel = SPI_DMA_CH_AUTO;
  _bus.config(bcfg);
  _panel.setBus(&_bus);

  auto pcfg = _panel.config();
  pcfg.pin_cs = 15;
  pcfg.pin_rst = 4;
  pcfg.panel_width = 240;
  pcfg.panel_height = 240;
  pcfg.invert = true;
  _panel.config(pcfg);

  setPanel(&_panel);
}

void setupPreviewBuffer()
{
  rgb565_buf = (uint16_t *)heap_caps_malloc(IMG_W * IMG_H * 2, MALLOC_CAP_SPIRAM);

  if (!rgb565_buf)
  {
    Serial.println("Không đủ RAM!");
    while (1);
  }
}

int drawMCU(JPEGDRAW *pDraw)
{
  tft.pushImageDMA(pDraw->x, pDraw->y, pDraw->iWidth, pDraw->iHeight, (uint16_t *)pDraw->pPixels);
  return 1;
}

void drawCenteredText(String text, int offsetY, int font, uint16_t fill) {
  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, fill);
  tft.drawString(text, tft.width()/2, tft.height()/2 + offsetY, font);
}

void showAnimations(const uint8_t frames[][512], int frameCount, String text, int font, uint16_t fill) {
  if (millis() - lastFrameTime < FRAME_DELAY) return;
  lastFrameTime = millis();

  int centerX = tft.width() / 2;
  int faceX = (tft.width() - FRAME_WIDTH) / 2;
  int faceY = (tft.height() - FRAME_HEIGHT) / 2 - 20;
  int textY = faceY + FRAME_HEIGHT + 20;

  // vẽ trong RAM
  sprite.fillSprite(fill);
  sprite.drawBitmap(0, 0, frames[frame], FRAME_WIDTH, FRAME_HEIGHT, TFT_WHITE);

  // đẩy ra màn hình
  sprite.pushSprite(faceX, faceY);

  // text
  tft.setTextDatum(MC_DATUM);
  tft.setTextColor(TFT_WHITE, fill);
  tft.drawString(text, centerX, textY, font);

  frame++;
  if (frame >= frameCount) {
    frame = 0;
  }
}