var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");

ctx.canvas.width  = 800;//window.innerWidth;
ctx.canvas.height = 400;//window.innerHeight;

ctx.textAlign = "left";
ctx.font = "16px Arial";

// building the graph

ctx.fillText("Historical Decisions",20,40); 
ctx.fillText("Future expectations",20,60); 
ctx.fillText("Observe current data",20,140);  
ctx.fillText("Choose an action",20,220); 
ctx.fillText("Observe loss (in future)",20,300); 

// drawing the example values

ctx.textAlign = "left";
ctx.font = "12px Arial";
ctx.fillStyle = "green";

ctx.fillText("Loss (previous supply temperature)",80,95);
ctx.fillText("Forecast",80,110); 
ctx.fillText("Room temperature",80,175); 
ctx.fillText("Set supply temperature",80,255);

// drawing the arrows

ctx.strokeStyle = "blue";
ctx.lineWidth = 2;

ctx.beginPath();
ctx.moveTo(70, 70);
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
ctx.lineTo(70, 200);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(75, 190);
ctx.lineTo(70, 200);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(65, 190);
ctx.lineTo(70, 200);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(70, 230);
ctx.lineTo(70, 280);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(75, 270);
ctx.lineTo(70, 280);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(65, 270);
ctx.lineTo(70, 280);
ctx.stroke();