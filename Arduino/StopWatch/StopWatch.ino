/*Simple LCD stopwatch program with stop, start, reset and lap buttons.*/

//including liblary for LCD
#include <LiquidCrystal.h> 

//setting up LCD INPUT pins
LiquidCrystal lcd(12,11,5,4,3,2);

//setting hours, minutes, secound and miliseconds to 0
int h=0;     
int m=0;     
int s=0;     
int ms=0;    

//defines pin for all buttons
const int start_pin = 8;    
const int stop1_pin = 9;    
const int reset_pin = 10;   
    
//defines starting points (in my case 0)
int start=0;     
int stop1=0;
int reset=0;


int brightness_pin = 6; //defines pin for setting brightness
int brightness=100; //you can change this number to change brightness 

void setup() 
{ 
  
  analogWrite(brightness_pin ,brightness); //this sets brightness on pin 6
  lcd.begin(16 ,2);  //starting LCD
   
  //defining pins if they are INPUT or OUTPUT pins
  pinMode(start_pin, INPUT);  
  pinMode(stop1_pin, INPUT);
  pinMode(reset_pin, INPUT);
  pinMode(brightness_pin, OUTPUT);
} 
void loop() 
{ 
  lcd.setCursor(0,1); 
  lcd.print("STOPWATCH");  
  lcd.setCursor(0,0);  
  lcd.print("TIME:"); 
  lcd.print(h); 
  lcd.print(":"); 
  lcd.print(m); 
  lcd.print(":"); 
  lcd.print(s);
  
 start = digitalRead(start_pin); //reading buton state
 if(start == HIGH) 
 {
  stopwatch();  //goes to sub program stopwatch
 }
 
} 



//--------------------SUB PROGRAMS-------------------------



void stopwatch()
{
  lcd.setCursor(0,0);   //setting start point on lcd 
  lcd.print("TIME:");   //writting TIME
  lcd.print(h);         //writing hours
  lcd.print(":");      
  lcd.print(m);         //writing minutes
  lcd.print(":"); 
  lcd.print(s);         //writing seconds
  ms=ms+10;           
  delay(10); 
   
 if(ms==590)           
  {
   lcd.clear();  //clears LCD
  }
  
 if(ms==590)     //if state for counting up seconds
  { 
  ms=0; 
  s=s+1; 
  }
  
  if(s==60)     //if state for counting up minutes
  { 
  s=0; 
  m=m+1; 
  }

  if(m==60)      //if state for counting up hours
  {  
  m=00; 
  h=h+01;  
  } 
   
  lcd.setCursor(0,1); 
  lcd.print("STOPWATCH");  

   stop1 = digitalRead(stop1_pin);  //reading buton state
 if(stop1 == HIGH)    //checking if button is pressed
 {
  stopwatch_stop();   //going to sub program
 }
  else
  {
   stopwatch();    //going to sub program
  }
}

void stopwatch_stop()
{
  lcd.setCursor(0,0); 
  lcd.print("TIME:"); 
  lcd.print(h); 
  lcd.print(":"); 
  lcd.print(m); 
  lcd.print(":"); 
  lcd.print(s);    
   
  lcd.setCursor(0,1); 
  lcd.print("STOPWATCH"); 

   start = digitalRead(start_pin);   //reading buton state
 if(start == HIGH)
 {
  stopwatch();    //going to sub program
 } 
 
 reset = digitalRead(reset_pin);   //reading buton state
 if(reset == HIGH)
 {
   stopwatch_reset();    //going to sub program
   loop();
  }
 if(reset == LOW)
 {
  stopwatch_stop();    //going to sub program
 }
}

void stopwatch_reset()
{
 lcd.clear();
 lcd.setCursor(0,1); 
 lcd.print("STOPWATCH");
 h=00;    //seting hours to 0
 m=00;    //seting minutes to 0
 s=00;    //seting seconds to 0
 return;  //exiting the program and returning to the point where entered the program
}
