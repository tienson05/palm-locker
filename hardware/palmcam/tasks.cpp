#include "tasks.h"
#include "config.h"

void sendTask(void *pv) {
  const TickType_t interval = pdMS_TO_TICKS(100);
  camera_fb_t *fb = NULL;

  while (true) {
    fb = NULL;
    xEventGroupWaitBits(ctrlEvent, EVT_RUN, pdFALSE, pdFALSE, portMAX_DELAY); // Đợi event

    if (xSemaphoreTake(fbMutex, portMAX_DELAY) == pdTRUE) {
      if (fb_shared) {
        fb = fb_shared;
        fb_shared = NULL;
      }
      xSemaphoreGive(fbMutex);
    }

    if (!fb) {
      vTaskDelay(interval);
      continue;
    }
    bool ok = webSocket.sendBIN(fb->buf, fb->len);
    Serial.printf("Send ok=%d\n", ok);
    esp_camera_fb_return(fb);
    fb = NULL;  // Reset sau khi return
    vTaskDelay(interval);
  }
}