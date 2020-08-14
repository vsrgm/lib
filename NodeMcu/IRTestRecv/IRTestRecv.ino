#include <IRrecv.h>

uint16_t RECV_PIN = D4;
IRrecv irrecv(RECV_PIN);

void setup() {
  Serial.begin(115200);

  // put your setup code here, to run once:
  irrecv.enableIRIn();  // Start the receiver

}

void loop() {
  decode_results results;
  unsigned long irhash = 0, ircode;
  // put your main code here, to run repeatedly:

  if (irrecv.decode(&results))
  {
    irhash = dump(&results);
    ircode = results.value;
    Serial.println(ircode);
    irrecv.resume();  // Receive the next value
  }
}

int compare(unsigned int oldval, unsigned int newval) {
  if (newval < oldval * .8) {
    return 0;
  }
  else if (oldval < newval * .8) {
    return 2;
  }
  else {
    return 1;
  }
}

#define USECPERTICK 2 //50
#define FNV_PRIME_32 16777619
#define FNV_BASIS_32 2166136261

unsigned long decodeHash(decode_results * results) {
  unsigned long hash = FNV_BASIS_32;
  for (int i = 1; i + 2 < results->rawlen; i++) {
    int value =  compare(results->rawbuf[i], results->rawbuf[i + 2]);
    // Add value into the hash
    hash = (hash * FNV_PRIME_32) ^ value;
  }
  return hash;
}

int c = 1;
unsigned long dump(decode_results *results)
{
  int count = results->rawlen;
  uint16 lbuf[200];
  Serial.println(c);
  c++;
  Serial.println("Hash: ");
  unsigned long hash = decodeHash(results);
  Serial.println(hash, HEX);
  Serial.println("For IR Scope/IrScrutinizer: ");
  for (int i = 1; i < count; i++) {

    if ((i % 2) == 1) {
      Serial.print("+");
      Serial.print(results->rawbuf[i]*USECPERTICK, DEC);
    }
    else {
      Serial.print(-(int)results->rawbuf[i]*USECPERTICK, DEC);
    }
    Serial.print(" ");
  }
  Serial.println("-127976");
  Serial.println("For Arduino sketch: ");
  Serial.print("unsigned int raw[");
  Serial.print(count, DEC);
  Serial.print("] = {");
  for (int i = 1; i < count; i++) {

    if ((i % 2) == 1) {
      Serial.print(results->rawbuf[i]*USECPERTICK, DEC);
    }
    else {
      Serial.print((int)results->rawbuf[i]*USECPERTICK, DEC);
    }
    Serial.print(",");
  }
  Serial.print("};");
  Serial.println("");
  Serial.print("irsend.sendRaw(raw,");
  Serial.print(count, DEC);
  Serial.print(",38);");
  Serial.println("");
  Serial.println("");
  return hash;
}
