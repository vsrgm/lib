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
#endif
