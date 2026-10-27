/*
  DHVANI EEG firmware - Arduino UNO R4 Minima + BioAmp EEG/EXG
  Developer: T.T

  Output (ONLY this goes on Serial, nothing else):  value1,value2\n   (one line = one sample)
  CONFIGURATION POINTS - verify against YOUR wiring before upload:
    - EEG_PINS: which analog pins your BioAmp OUT wires go to (default A0, A1)
    - NUM_CHANNELS: must match "Channels" in DHVANI
    - SAMPLE_RATE: must match "Sampling rate" in DHVANI
    - BAUD: must match "Baud" in DHVANI
  Power the BioAmp per its own datasheet (battery recommended, no mains-connected laptop
  charger while electrodes are on a person). Output is raw ADC counts.
*/
#define NUM_CHANNELS 2
#define SAMPLE_RATE  250        // Hz
#define BAUD         115200
#define ADC_BITS     14         // R4 Minima supports up to 14-bit; ADC max = 16383

const uint8_t EEG_PINS[NUM_CHANNELS] = {A0, A1};
const unsigned long PERIOD_US = 1000000UL / SAMPLE_RATE;
unsigned long nextTick;

void setup() {
  Serial.begin(BAUD);
  analogReadResolution(ADC_BITS);
  nextTick = micros();
}

void loop() {
  unsigned long now = micros();
  if ((long)(now - nextTick) < 0) return;   // wait for next sample slot
  nextTick += PERIOD_US;
  for (uint8_t c = 0; c < NUM_CHANNELS; c++) {
    Serial.print(analogRead(EEG_PINS[c]));
    Serial.print(c < NUM_CHANNELS - 1 ? ',' : '\n');
  }
}
