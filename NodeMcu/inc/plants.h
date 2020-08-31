#ifndef __PLANTS
#define __PLANTS
enum eeromoffset
{
  MAGIC = 1,
  BCNT,
  BTOTAL,
  TCNT,
  TTOTAL,
};

enum feeds
{
  BPLANTS = 0,
  TPLANTS,
  RESETREPORT,
  REPORT,
  GETREPORT,
  AUTOUPDATETEMPHUMID,
  RESETDATA,
};

struct dateformat
{
  char month[12];
  int date;
  int year;
  int timehour;
  int timemin;
  char meridies[3];
};

struct feedinput
{
  int type;
  int value1;
  int value2;
  int value3;
  int value4;
  struct dateformat date;
};

#endif
