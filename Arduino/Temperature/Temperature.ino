
const int lm35_pin = A1;  /* LM35 O/P pin */
#define ENABLE1 2
#define ENABLE2 3
#define ENABLE3 4
#define SEG1 5
#define SEG2 6
#define SEG3 7
#define SEG4 8
#define SEG5 9
#define SEG6 10
#define SEG7 11
#define SEG8 12
#define AVERAGEDELAYCOUNT 600
struct 
{
  bool seg[8];
}val[] = 
{
  {LOW, LOW, LOW, LOW, LOW, HIGH, LOW, HIGH},     // 0
  {HIGH, HIGH, LOW, HIGH, HIGH, HIGH, LOW, HIGH}, // 1
  {HIGH, LOW, LOW, LOW, LOW, HIGH, HIGH, LOW},    // 2
  {HIGH, LOW, LOW, HIGH, LOW, HIGH, LOW, LOW},    // 3
  {LOW, HIGH, LOW, HIGH, HIGH, HIGH, LOW, LOW},   // 4
  {LOW, LOW, HIGH, HIGH, LOW, HIGH, LOW, LOW},    // 5
  {LOW, LOW, HIGH, LOW, LOW, HIGH, LOW, LOW},     // 6
  {HIGH, LOW, LOW, HIGH, HIGH, HIGH, LOW, HIGH},  // 7
  {LOW, LOW, LOW, LOW, LOW, HIGH, LOW, LOW},      // 8
  {LOW, LOW, LOW, HIGH, LOW, HIGH, LOW, LOW},     // 9
  {HIGH, HIGH, HIGH, HIGH, HIGH, LOW, HIGH, HIGH},// .
  {LOW, LOW, LOW, LOW, LOW, LOW, LOW, HIGH},     // 0
  {HIGH, HIGH, LOW, HIGH, HIGH, LOW, LOW, HIGH}, // 1
  {HIGH, LOW, LOW, LOW, LOW, LOW, HIGH, LOW},    // 2
  {HIGH, LOW, LOW, HIGH, LOW, LOW, LOW, LOW},    // 3
  {LOW, HIGH, LOW, HIGH, HIGH, LOW, LOW, LOW},   // 4
  {LOW, LOW, HIGH, HIGH, LOW, LOW, LOW, LOW},    // 5
  {LOW, LOW, HIGH, LOW, LOW, LOW, LOW, LOW},     // 6
  {HIGH, LOW, LOW, HIGH, HIGH, LOW, LOW, HIGH},  // 7
  {LOW, LOW, LOW, LOW, LOW, LOW, LOW, LOW},      // 8
  {LOW, LOW, LOW, HIGH, LOW, LOW, LOW, LOW},     // 9
};

void setup()
{
  Serial.begin(115200);
  pinMode(ENABLE1, OUTPUT);
  pinMode(ENABLE2, OUTPUT);
  pinMode(ENABLE3, OUTPUT);
  
  pinMode(SEG1, OUTPUT);
  pinMode(SEG2, OUTPUT);
  pinMode(SEG3, OUTPUT);
  pinMode(SEG4, OUTPUT);
  pinMode(SEG5, OUTPUT);
  pinMode(SEG6, OUTPUT);
  pinMode(SEG7, OUTPUT);
  pinMode(SEG8, OUTPUT);
}

long readVcc()
{
  // Read 1.1V reference against AVcc
  // set the reference to Vcc and the measurement to the internal 1.1V reference
  #if defined(__AVR_ATmega32U4__) || defined(__AVR_ATmega1280__) || defined(__AVR_ATmega2560__)
    ADMUX = _BV(REFS0) | _BV(MUX4) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
  #elif defined (__AVR_ATtiny24__) || defined(__AVR_ATtiny44__) || defined(__AVR_ATtiny84__)
     ADMUX = _BV(MUX5) | _BV(MUX0) ;
  #else
    ADMUX = _BV(REFS0) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
  #endif  
 
  delay(2); // Wait for Vref to settle
  ADCSRA |= _BV(ADSC); // Start conversion
  while (bit_is_set(ADCSRA,ADSC)); // measuring
 
  uint8_t low  = ADCL; // must read ADCL first - it then locks ADCH  
  uint8_t high = ADCH; // unlocks both
 
  long result = (high<<8) | low;
 
  result = 1125300L / result; // Calculate Vcc (in mV); 1125300 = 1.1*1023*1000
  return result; // Vcc in millivolts
}

void loop()
{
  unsigned char ch;
  int temp_adc_val;
  float temp_val;
  static float temp_avg = 0;
  static int led_write = 0;
  int a, b, c;
  float refV;
  static int count = AVERAGEDELAYCOUNT;  


  refV = readVcc();
  temp_adc_val = analogRead(lm35_pin);  /* Read Temperature */
  temp_val = (temp_adc_val * refV/1024); /* Convert adc value to equivalent voltage */
  temp_val = (temp_val / 10); /* LM35 gives output of 10mv/°C */  
  temp_avg = temp_avg + temp_val;
  if (count == 0)
  {
    count = AVERAGEDELAYCOUNT;
    led_write = temp_avg/AVERAGEDELAYCOUNT*10;
    temp_avg = 0;
  }

  if (led_write == 0)
  {
    led_write = temp_val * 10;
  }
  
  count--;
  Serial.print("Temperature = ");
  Serial.print(temp_val);
  Serial.print(" Degree Celsius\r\n");
  a = led_write/100;
  b = (led_write - (a*100))/10;
  c = (led_write - (a*100) - (b*10));

  delay(5);
  a = led_write/100;
  digitalWrite(ENABLE1, HIGH);
  digitalWrite(ENABLE2, LOW);
  digitalWrite(ENABLE3, LOW);

  for (int i=0; i < 8; i++)
  {
    digitalWrite(SEG1+i, val[a].seg[i]);
  }

  delay(5);
  b = (led_write - (a*100))/10;
  
  digitalWrite(ENABLE1, LOW);
  digitalWrite(ENABLE2, LOW);
  digitalWrite(ENABLE3, HIGH);
  for (int i=0; i < 8; i++)
  {
    digitalWrite(SEG1+i, val[11+b].seg[i]);
  }
  delay(5);
  c = (led_write - (a*100) - (b*10));
  digitalWrite(ENABLE1, LOW);
  digitalWrite(ENABLE3, LOW);
  digitalWrite(ENABLE2, HIGH);
  for (int i=0; i < 8; i++)
  {
    digitalWrite(SEG1+i, val[c].seg[i]);
  }
}
