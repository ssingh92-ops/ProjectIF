#include "BluetoothSerial.h"

#define RXD2 16  // Pump TX → ESP32 RX
#define TXD2 17  // Pump RX ← ESP32 TX

HardwareSerial pumpSerial(2); // UART2 for pump

BluetoothSerial SerialBT; // Bluetooth serial connection

String incomingCommand = "";

void setup() {
  Serial.begin(115200);   // USB Serial Monitor
  SerialBT.begin("BTPump");  // Bluetooth device name

  pumpSerial.begin(9600, SERIAL_8N1, RXD2, TXD2); // EZO-PMP UART
  delay(3000); // Allow the pump to boot

  Serial.println("EZO-PMP Bluetooth UART communication initialized.");
  Serial.println("You can now send commands over Bluetooth.");
}

void loop() {
  // ==== 1. Handle input from Bluetooth terminal ====
  if (SerialBT.available()) {
    incomingCommand = SerialBT.readStringUntil('\n');
    incomingCommand.trim(); // Remove whitespace

    if (incomingCommand.length() > 0) {
      // Send Bluetooth command to pump with carriage return
      pumpSerial.print(incomingCommand + "\r");

      SerialBT.print("[Sent to Pump via BT]: ");
      SerialBT.println(incomingCommand);
    }
  }

  // ==== 2. Handle pump responses and send them back via Bluetooth ====
  static String pumpResponse = "";
  while (pumpSerial.available()) {
    char c = pumpSerial.read();

    if (c == '\r') {  // End of message
      SerialBT.print("[Pump Response]: ");
      SerialBT.println(pumpResponse);
      pumpResponse = ""; // Reset buffer
    } else {
      pumpResponse += c;
    }
  }

  delay(20);  // Small delay to avoid serial buffer overflow
}