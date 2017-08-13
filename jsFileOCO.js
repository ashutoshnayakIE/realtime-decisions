var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");

ctx.canvas.width  = 800;//window.innerWidth;
ctx.canvas.height = 400;//window.innerHeight;

ctx.textAlign = "left";
ctx.font = "16px Arial";

// building the graph

ctx.fillText("Historical Decisions",20,40);  
ctx.fillText("Choose an action",20,120);  
ctx.fillText("Observe current data",20,200); 
ctx.fillText("Observe loss",20,280); 

// drawing the example values

ctx.textAlign = "left";
ctx.font = "12px Arial";
ctx.fillStyle = "green";

ctx.fillText("Number of packets in queue",80,75);
ctx.fillText("Packet selection",80,155); 
ctx.fillText("Bandwidth cost function",80,235);

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