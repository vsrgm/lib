
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

void loop()
{
  unsigned char ch;
  int temp_adc_val;
  float temp_val;
  int led_write;
  int a, b, c;
  temp_adc_val = analogRead(lm35_pin);  /* Read Temperature */
  temp_val = (temp_adc_val * 4.88); /* Convert adc value to equivalent voltage */
  led_write = temp_val;
  temp_val = (temp_val / 10); /* LM35 gives output of 10mv/°C */
#if 1
  Serial.print("Temperature = ");
  Serial.print(temp_val);
  Serial.print(" Degree Celsius");
  Serial.print(" led_write = ");  
  Serial.print(led_write);
  Serial.print(" 1st = ");
  a = led_write/100;
  Serial.print(a);

  Serial.print(" 2nd = ");
  b = (led_write - (a*100))/10;
  Serial.print(b);
  

  Serial.print(" 3rd = ");
  c = (led_write - (a*100) - (b*10));
  Serial.print(c);
  Serial.print("\r\n");
#endif
  delay(15);
  a = led_write/100;
  digitalWrite(ENABLE1, HIGH);
  digitalWrite(ENABLE2, LOW);
  digitalWrite(ENABLE3, LOW);

  for (int i=0; i < 8; i++)
  {
    digitalWrite(SEG1+i, val[a].seg[i]);
  }

  delay(15);
  b = (led_write - (a*100))/10;
  
  digitalWrite(ENABLE1, LOW);
  digitalWrite(ENABLE2, LOW);
  digitalWrite(ENABLE3, HIGH);
  for (int i=0; i < 8; i++)
  {
    digitalWrite(SEG1+i, val[11+b].seg[i]);
  }
  delay(15);
  c = (led_write - (a*100) - (b*10));
  digitalWrite(ENABLE1, LOW);
  digitalWrite(ENABLE3, LOW);
  digitalWrite(ENABLE2, HIGH);
  for (int i=0; i < 8; i++)
  {
    digitalWrite(SEG1+i, val[c].seg[i]);
  }
}
