#include "config.h"
#include "camera_module.h"
#include "tft_display.h"
#include "websocket_module.h"
#include "tasks.h"
#include "animations.h"

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println(WiFi.localIP());

  tft.init();
  tft.setRotation(0);
  tft.setSwapBytes(true);
  tft.initDMA();
  sprite.createSprite(64, 64);

  setupPreviewBuffer();
  initCamera();
  initWebSocket();

  fbMutex = xSemaphoreCreateMutex();
  ctrlEvent = xEventGroupCreate();
  xTaskCreatePinnedToCore(sendTask, "send", 8192, NULL, 2, &sendTaskHandle, 0);
}

void loop() {
  webSocket.loop();

  if (needCleanup && millis() - cleanupTime > 100) {
    if (xSemaphoreTake(fbMutex, portMAX_DELAY) == pdTRUE) {
      if (fb_shared != NULL) {
        esp_camera_fb_return(fb_shared);
        fb_shared = NULL;
      }
      xSemaphoreGive(fbMutex);
    }
    needCleanup = false;
  }

  switch (currentState) {
    case STATE_CONNECTED:
      if(!needRedraw) {
        tft.fillScreen(TFT_GREEN);
        drawCenteredText("SYSTEM READY", 0, 4, TFT_GREEN);
        needRedraw = true;
      }

      if (millis() - stateTime > 2000) {
        currentState = STATE_IDLE;
        needRedraw = false;
      }
      return;

    case STATE_SUCCESS:
      if(!needRedraw) {
        tft.fillScreen(TFT_GREEN);
        needRedraw = true;
      }
      showAnimations(frames_success, FRAME_COUNT_SUCCESS, "SUCCESS", 4, TFT_GREEN);

      if (millis() - stateTime > 4000) {
        currentState = STATE_IDLE;
        needRedraw = false;
      }
      return;

    case STATE_FAIL:
      if(needRedraw) {
        tft.fillScreen(TFT_RED);
        needRedraw = false;
      }
      showAnimations(frames_fail, FRAME_COUNT_FAIL, "TRY AGAIN", 4, TFT_RED);

      if (millis() - stateTime > 3000) {
        currentState = STATE_IDLE;
      }
      return;

    case STATE_FULL:
      if(needRedraw) {
        tft.fillScreen(TFT_RED);
        needRedraw = false;
      }
      showAnimations(frames_fail, FRAME_COUNT_FAIL, "NO SLOT", 4, TFT_RED);
      return;

    case STATE_DISCONNECTED:
      if(needRedraw) {
        tft.fillScreen(TFT_RED);
        drawCenteredText("DISCONNECTED", 0, 4, TFT_RED);
        needRedraw = false;
      }
      return;

    case STATE_WAIT:
      if(!needRedraw) {
        tft.fillScreen(TFT_GREEN);
      }
      showAnimations(frames_wait, FRAME_COUNT_WAIT, "WAITING", 4, TFT_GREEN);
      return;

    case STATE_IDLE:
      if(!needRedraw) {
        tft.fillScreen(TFT_GREEN);
        needRedraw = true;
      }
      showAnimations(frames_welcome, FRAME_COUNT_WELCOME, "WELCOME", 4, TFT_GREEN);
      return;
    
    case STATE_SEND_TAKE:
      break;
  }

  if (!webSocket.isConnected())
    return;

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb)
    return;

  if (xSemaphoreTake(fbMutex, 0) == pdTRUE) {
    if (fb_shared)
      esp_camera_fb_return(fb_shared);
    fb_shared = fb;
    xSemaphoreGive(fbMutex);
  } else {
    esp_camera_fb_return(fb);
    return;
  }

  jpeg.openRAM(fb->buf, fb->len, drawMCU);
  jpeg.setPixelType(RGB565_LITTLE_ENDIAN);
  tft.startWrite();
  jpeg.decode(20, 45, JPEG_SCALE_QUARTER); // SVGA 
  tft.endWrite();
  jpeg.close();
  currentState = STATE_IDLE;
}