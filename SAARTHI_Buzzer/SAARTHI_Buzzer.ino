const int BUZZER_PIN = 23;

String command = "";

bool buzzerActive = false;
unsigned long buzzerStartTime = 0;

const unsigned long BEEP_DURATION = 200;

void setup() {
  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  command.reserve(32);

  Serial.println("SAARTHI READY");
  Serial.println("BUZZER OFF");
}

void loop() {

  // Read commands from the computer
  while (Serial.available() > 0) {
    char incoming = Serial.read();

    if (incoming == '\n') {
      command.trim();

      if (command == "ALERT" && !buzzerActive) {
        digitalWrite(BUZZER_PIN, HIGH);

        buzzerActive = true;
        buzzerStartTime = millis();

        Serial.println("ALERT RECEIVED");
      }

      command = "";
    }
    else if (incoming != '\r' && command.length() < 32) {
      command += incoming;
    }
  }

  // Stop the buzzer automatically
  if (buzzerActive &&
      millis() - buzzerStartTime >= BEEP_DURATION) {

    digitalWrite(BUZZER_PIN, LOW);
    buzzerActive = false;

    Serial.println("BUZZER OFF");
  }
}