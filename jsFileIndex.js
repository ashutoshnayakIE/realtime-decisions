var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");

ctx.canvas.width  = 800;//window.innerWidth;
ctx.canvas.height = 400;//window.innerHeight;

ctx.textAlign = "left";

ctx.font = "20px Arial";

var dataBox = document.createElement("TEXTAREA");
ctx.fillText("Data",0,20); 
var constraintBox = document.createElement("TEXTAREA");
ctx.fillText("Constraints",0,140); 
var regretBox = document.createElement("TEXTAREA");
ctx.fillText("Regret",0,220); 
var actionBox = document.createElement("TEXTAREA");
ctx.fillText("Action based on",0,300); 
/*
// drawing the lines 
ctx.beginPath();
ctx.moveTo(0, 100);
ctx.lineTo(550, 100);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(0, 180);
ctx.lineTo(550, 180);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(0, 260);
ctx.lineTo(550, 260);
ctx.stroke();
*/
// ---------------------------------------------------------------
ctx.font = "16px Arial";
ctx.fillStyle = "green";

var hostoricalBox = document.createElement("TEXTAREA");
ctx.fillText("Historical",0,50); 
var currentBox = document.createElement("TEXTAREA");
ctx.fillText("Current",200,50); 
var futureBox = document.createElement("TEXTAREA");
ctx.fillText("Future",400,50); 

// ---------------------------------------------------------------

ctx.fillStyle = "gray";

var observationsBox = document.createElement("TEXTAREA");
ctx.fillText("Observations",0,70); 
var decisionBox = document.createElement("TEXTAREA");
ctx.fillText("Decisions",0,90); 
var observationBox = document.createElement("TEXTAREA");
ctx.fillText("Observation",200,70); 
var expectedBox = document.createElement("TEXTAREA");
ctx.fillText("Expected observation",400,70); 
var shorttermBox = document.createElement("TEXTAREA");
ctx.fillText("Short term",0,170); 
var longtermBox = document.createElement("TEXTAREA");
ctx.fillText("Long term",200,170); 
var staticBox = document.createElement("TEXTAREA");
ctx.fillText("Static",0,250); 
var dynamicBox = document.createElement("TEXTAREA");
dynamicBox.name = 'dynamidjbcljkdbcBox'
ctx.fillText("Dynamic",200,250); 
var knownBox = document.createElement("TEXTAREA");
ctx.fillText("Known regret function",0,330); 
var unknownBox = document.createElement("TEXTAREA");
ctx.fillText("Unknown regret function",200,330); 

function changeSL() {
	document.getElementById("dynamicBox").value = "Fifth Avenue, New York City"; 

	var canvas = document.getElementById("myCanvas");
    var x = document.createElement("TEXTAREA");
    var ctx = canvas.getContext("2d");
    //ctx.fillText("textArea.value", 40, 60);
}
/*
function changeSL() {
    var x = document.getElementById("sl");
    ctx.fillStyle = 'red';
    console.log("huhaaaa")
}

function changeOL() {
    var x = document.getElementById("ol");
    ctx.fillStyle = 'red';
}

function changeOCO() {
    var x = document.getElementById("oco");
    ctx.fillStyle = 'red';
}

function changeMPC() {
    var x = document.getElementById("mpc");
    ctx.fillStyle = 'red';
}

function changeOSO() {
    var x = document.getElementById("oso");
    ctx.fillStyle = 'red';
}
*/