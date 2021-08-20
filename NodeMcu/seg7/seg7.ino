uint8_t pinA = D0;
uint8_t pinB = D1;
uint8_t pinC = D2;
uint8_t pinD = D4;
uint8_t pinE = D5;
uint8_t pinF = D6;
uint8_t pinG = D7;
uint8_t pinDT = D3;
uint8_t pinD8 = D8;
uint8_t pinD9 = D9;
uint8_t pinD10 = D10;

uint8_t pinarray[] =
{pinA, pinB, pinC, pinD, pinE, pinF, pinG};

uint8_t pinarrayval[][7] = {
  { LOW, LOW, LOW, LOW, LOW, LOW, HIGH}, // 0
  {HIGH, LOW, LOW, HIGH, HIGH, HIGH, HIGH}, // 1
  { LOW, LOW, HIGH, LOW, LOW, HIGH, LOW}, // 2
  { LOW, LOW, LOW, LOW, HIGH, HIGH, LOW}, // 3
  {HIGH, LOW, LOW, HIGH, HIGH, LOW, LOW}, // 4
  { LOW, HIGH, LOW, LOW, HIGH, LOW, LOW}, // 5
  { LOW, HIGH, LOW, LOW, LOW, LOW, LOW}, // 6
  { LOW, LOW, LOW, HIGH, HIGH, HIGH, HIGH}, // 7
  { LOW, LOW, LOW, LOW, LOW, LOW, LOW}, // 8
  { LOW, LOW, LOW, LOW, HIGH, LOW, LOW}, // 9
  {HIGH, HIGH, HIGH, HIGH, HIGH, HIGH, HIGH}, // clear
  { LOW,  LOW,  LOW,  LOW,  LOW,  LOW,  LOW}, // clear

};


void setup() {
  pinMode(pinA, OUTPUT);
  pinMode(pinB, OUTPUT);
  pinMode(pinC, OUTPUT);
  pinMode(pinD, OUTPUT);
  pinMode(pinE, OUTPUT);
  pinMode(pinF, OUTPUT);
  pinMode(pinG, OUTPUT);
  pinMode(pinDT, OUTPUT);

  pinMode(pinD8, OUTPUT);
  pinMode(pinD9, OUTPUT);
  pinMode(pinD10, OUTPUT);
  for (int i = 0 ; i < sizeof(pinarray); i++)
    digitalWrite(pinarray[i], pinarrayval[11][i]);

}

void toogle(bool enable)
{
  static int idx = 0;
  static int count = 0;
  count++;
  static int number = 0;
  if (count == 20000)
  {
    number += 1;
    count = 0;
  }

  if (number == 10)
    number = 0;

  for (int i = 0 ; i < sizeof(pinarray); i++)
    for (int j = 0 ; j < sizeof(pinarray); j++)
    {
      if ((enable) & (i == j))
      {
        digitalWrite(pinarray[i], !pinarrayval[number][i]);
      } else
        digitalWrite(pinarray[j], !pinarrayval[10][j]);

    }
#if 0
  if ((i == idx) && enable)
  {
    digitalWrite(pinarray[i], LOW);
  }
  else
  {
    digitalWrite(pinarray[i], HIGH);

  }
}
#endif

idx++;
if (idx == sizeof(pinarray))
{
  idx = 0;
}

}
void loop()
{
#if 0
  digitalWrite(pinA, LOW);
  delay(1000);
  digitalWrite(pinA, HIGH);
  delay(1000);

#else
  while (true)
  {
    toogle(true);
    toogle(false);
    toogle(false);
    toogle(false);
    toogle(false);
    toogle(false);
    toogle(false);
    toogle(false);
    toogle(false);

    delay(1);
    //delayMicroseconds(1000);

  }
#endif
}
