#include <sys/_types.h>
#include "config.h"

// const char *ssid = "VAN THANH";
// const char *password = "ad123456";
const char *ssid = "Xom Tro";
const char *password = "12346789";

// const char *ssid = "BT";
// const char *password = "toan12345";

// String serverName = "192.168.101.45";
// const int serverPort = 5000;
// String serverName = "172.20.10.8";
// const int serverPort = 5000;
String serverName = "192.168.1.244";
const int serverPort = 8000;

WebSocketsClient webSocket;

camera_fb_t *fb_shared = NULL;
SemaphoreHandle_t fbMutex;
EventGroupHandle_t ctrlEvent;
TaskHandle_t sendTaskHandle = NULL;

uint16_t *rgb565_buf = NULL;

unsigned long stateTime = 0;
bool needRedraw = false;
ScreenState currentState = STATE_IDLE;

bool needCleanup = false;
unsigned long cleanupTime = 0;