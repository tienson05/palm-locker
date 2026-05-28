#include <Arduino.h>
#include "htcl.h"

// ====== PIN CONFIG ======
#define SEND_IR 7
#define TAKE_IR 8

// ====== STATE (lưu trạng thái trước đó) ======
bool lastSendState = HIGH;
bool lastTakeState = HIGH;

// ====== CHỐNG SPAM (cooldown) ======
unsigned long lastSendTime = 0;
unsigned long lastTakeTime = 0;
const unsigned long cooldown = 2000; // ms

// ====== SETUP PIN ======
void setupButtons()
{
  pinMode(SEND_IR, INPUT);
  pinMode(TAKE_IR, INPUT);
}

// ====== HANDLE IR (CALL TRONG LOOP) ======
void handleButtons()
{
  unsigned long now = millis();

  bool currentSend = digitalRead(SEND_IR);
  bool currentTake = digitalRead(TAKE_IR);

  // ===== SEND: HIGH -> LOW (có xe) =====
  if (lastSendState == HIGH && currentSend == LOW && (now - lastSendTime > cooldown))
  {
    lastSendTime = now;
    sendCommand("open");
    Serial.println("SEND detected (1 lần)");
  }

  // ===== TAKE: HIGH -> LOW (có xe) =====
  if (lastTakeState == HIGH && currentTake == LOW && (now - lastTakeTime > cooldown))
  {
    lastTakeTime = now;
    sendCommand("close");
    Serial.println("TAKE detected (1 lần)");
  }

  // cập nhật trạng thái cũ
  lastSendState = currentSend;
  lastTakeState = currentTake;
}