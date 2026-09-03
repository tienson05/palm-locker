#include "camera_module.h"

#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27

#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

void initCamera() {
  camera_config_t config;

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;

  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_SVGA; // 640 × 480 FRAMESIZE_SVGA 800x600, FRAMESIZE_VGA
  config.jpeg_quality = 10;
  config.fb_count = 3;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed 0x%x", err);
    delay(1000);
    ESP.restart();
  }

  sensor_t *s = esp_camera_sensor_get();

   // Tối ưu cho độ nét
  s->set_vflip(s, 1);        // Lật dọc nếu cần
  s->set_hmirror(s, 1);      // Lật ngang nếu cần
  s->set_brightness(s, 0);    // Độ sáng ( -2 to 2 )
  s->set_contrast(s, 2);      // Độ tương phản ( -2 to 2 ) - tăng lên
  s->set_saturation(s, 0);    // Độ bão hòa ( -2 to 2 )
  s->set_sharpness(s, 3);     // Độ nét ( 0 to 3 ) - QUAN TRỌNG: tối đa
  
  s->set_gain_ctrl(s, 0);     // Tắt AGC
  s->set_exposure_ctrl(s, 0); // Tắt AEC
  s->set_awb_gain(s, 1);      // Bật AWB
  
  s->set_aec_value(s, 100);   // Exposure value
  s->set_agc_gain(s, 16);     // Gain
  s->set_whitebal(s, 1);      // White balance
  
  // Điều chỉnh cho môi trường ánh sáng yếu
  s->set_gainceiling(s, GAINCEILING_4X); // Tăng gain ceiling
}