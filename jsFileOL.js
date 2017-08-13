var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");

ctx.canvas.width  = 800;//window.innerWidth;
ctx.canvas.height = 400;//window.innerHeight;

ctx.textAlign = "left";
ctx.font = "16px Arial";

// building the graph

ctx.fillText("Historical Observations",20,40); 
ctx.fillText("Choose an action",20,120);  
ctx.fillText("Observe current data",20,200); 
ctx.fillText("Observe loss",20,280); 
ctx.fillText("Update Action",20,360); 

// drawing the example values

ctx.textAlign = "left";
ctx.font = "12px Arial";
ctx.fillStyle = "green";

ctx.fillText("(X,Y)",80,75); 
ctx.fillText("regression coefficients",80,155); 
ctx.fillText("square loss function",80,235); 
ctx.fillText("update weights",80,315);

// drawing the arrows

ctx.strokeStyle = "blue";
ctx.lineWidth = 2;

ctx.beginPath();
ctx.moveTo(70, 50);
ctx.lineTo(70, 100);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(75, 90);
ctx.lineTo(70, 100);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(65, 90);
ctx.lineTo(70, 100);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(70, 130);
ctx.lineTo(70, 180);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(75, 170);
ctx.lineTo(70, 180);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(65, 170);
ctx.lineTo(70, 180);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(70, 210);
ctx.lineTo(70, 260);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(75, 250);
ctx.lineTo(70, 260);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(65, 250);
ctx.lineTo(70, 260);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(70, 290);
ctx.lineTo(70, 340);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(75, 330);
ctx.lineTo(70, 340);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(65, 330);
ctx.lineTo(70, 340);
ctx.stroke();