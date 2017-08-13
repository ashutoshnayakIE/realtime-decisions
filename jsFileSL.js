var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");

ctx.canvas.width  = 800;//window.innerWidth;
ctx.canvas.height = 400;//window.innerHeight;

ctx.textAlign = "left";
ctx.font = "16px Arial";

// building the graph

ctx.fillText("Historical Observations",20,40); 
ctx.fillText("Choose an action",20,140);  
ctx.fillText("Observe current data",20,240); 
ctx.fillText("Observe loss",20,340); 

// drawing the example values

ctx.textAlign = "left";
ctx.font = "12px Arial";
ctx.fillStyle = "green";

ctx.fillText("(X,Y)",80,85); 
ctx.fillText("regression coefficients",80,190); 
ctx.fillText("square loss function",80,290); 

// drawing the arrows

ctx.strokeStyle = "blue";
ctx.lineWidth = 2;

ctx.beginPath();
ctx.moveTo(70, 50);
ctx.lineTo(70, 120);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(75, 110);
ctx.lineTo(70, 120);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(65, 110);
ctx.lineTo(70, 120);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(70, 150);
ctx.lineTo(70, 220);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(75, 210);
ctx.lineTo(70, 220);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(65, 210);
ctx.lineTo(70, 220);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(70, 250);
ctx.lineTo(70, 320);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(75, 310);
ctx.lineTo(70, 320);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(65, 310);
ctx.lineTo(70, 320);
ctx.stroke();