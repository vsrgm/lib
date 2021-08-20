var SS = SpreadsheetApp.openByUrl("https://docs.google.com/spreadsheets/d/1cpEsRZYk61c78nDs-yTntMRSB8xoi8AeRr6t6UO0TRc/edit#gid=0");
var sheet = SS.getSheetByName('Temp1');
var str = "";
var readvalue = "";

function onOpen(){
	var ui = SpreadsheetApp.getUi();
	ui.createMenu('ESP8266_Temp_Logger')
		.addItem('Clear', 'Clear')
		.addToUi();
}

function Clear(){
	sheet.getRange("C10").setValue("Hello");
}

function doPost(e) {

	var parsedData;
	var result = {};

	try { 
		parsedData = JSON.parse(e.postData.contents);
	} 
	catch(f){
		return ContentService.createTextOutput("Error in parsing request body: " + f.message);
	}

	if (parsedData !== undefined){
		// Common items first
		// data format: 0 = display value(literal), 1 = object value
		var flag = parsedData.format;

		if (flag === undefined){
			flag = 0;
		}

		switch (parsedData.command) {
			case "read":
				var tmp = SS.getSheetByName(parsedData.sheet_name);
				readvalue = parsedData.values.split(",");
				var cell = tmp.getRange('D2'); 
				cell.setValue(readvalue);
				SpreadsheetApp.flush();

				break;

			case "cell":
				var tmp = SS.getSheetByName(parsedData.sheet_name);
				var nextFreeRow = tmp.getLastRow() + 1;
				var dataArr = parsedData.values.split(",");      
				str = "Success";
				var cell = tmp.getRange(dataArr[0]); 
				cell.setValue(dataArr[1]);
				SpreadsheetApp.flush();
				break;

			case "appendRow":
				var tmp = SS.getSheetByName(parsedData.sheet_name);
				var nextFreeRow = tmp.getLastRow() + 1;
				var dataArr = parsedData.values.split(",");         
				tmp.appendRow(dataArr);

				str = "Success";
				SpreadsheetApp.flush();   
				break;
		}    
		return ContentService.createTextOutput(str);
	} // endif (parsedData !== undefined)

	else{
		return ContentService.createTextOutput("Error! Request body empty or in incorrect format.");
	}  
}

function doGet(e){
	var read = e.parameter.read;

	if (read !== undefined){
		return ContentService.createTextOutput(sheet.getRange(sheet.getRange('D2').getValue()).getValue());
	}
}
