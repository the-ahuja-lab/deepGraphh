


window.addEventListener("load", initslider);
var progress1 = document.getElementById("stp1");

function initslider(){
  var slider = document.getElementById("myRange");
  var output = document.getElementById("demo");
  
  var slider1 = document.getElementById("myRange1");
  var output1 = document.getElementById("demo1");
  
  var slider2 = document.getElementById("myRange2");
  var output2 = document.getElementById("demo2");

  
  output.innerHTML = slider.value;
  
  slider.oninput = function() {
    output.innerHTML = this.value;
  }
  
  
  output1.innerHTML = slider1.value;
  
  slider1.oninput = function() {
    output1.innerHTML = this.value;
  }
  
  
  output2.innerHTML = slider2.value;
  
  slider2.oninput = function() {
    output2.innerHTML = this.value;
  }
}

function submit_F(){
  event.preventDefault();
  var csv=document.getElementById("csv").files[0];
  var fl=false;
  if(csv==null){
    console.log(csv)
    console.log("Empty");
  }
  else{
    console.log(csv);
    fl=true;
    console.log(typeof(csv))
  }  
  if(fl==true){
     console.log("here");
     progress1.style.backgroundColor='green';
    }
}

function initslider(){
  var slider = document.getElementById("myRange");
  var output = document.getElementById("demo");

  var slider1 = document.getElementById("myRange1");
  var output1 = document.getElementById("demo1");

  var slider2 = document.getElementById("myRange2");
  var output2 = document.getElementById("demo2");

  var slider3 = document.getElementById("myRange3");
  var output3 = document.getElementById("demo3");

  var slider4 = document.getElementById("myRange4");
  var output4 = document.getElementById("demo4");

  var slider5 = document.getElementById("num_ts");
  var output5 = document.getElementById("demo5");

  var slider6 = document.getElementById("graph_ft");
  var output6 = document.getElementById("demo6");

  var slider7 = document.getElementById("bond_ft");
  var output7 = document.getElementById("bondft");

  output.innerHTML = slider.value;

  slider.oninput = function() {
    output.innerHTML = this.value;
  }


  output1.innerHTML = slider1.value;

  slider1.oninput = function() {
    output1.innerHTML = this.value;
  }


  output2.innerHTML = slider2.value;

  slider2.oninput = function() {
    output2.innerHTML = this.value;
  }

  output3.innerHTML = slider3.value;
  slider3.oninput = function() {
    output3.innerHTML = this.value;
  }

  output4.innerHTML = slider4.value;
  slider4.oninput = function() {
    output4.innerHTML = this.value;
  }

  output5.innerHTML = slider5.value;
  slider5.oninput = function() {
    output5.innerHTML = this.value;
  }

  output6.innerHTML = slider6.value;
  slider6.oninput = function() {
    output6.innerHTML = this.value;
  }

  output7.innerHTML = slider7.value;
  slider7.oninput = function() {
    output7.innerHTML = this.value;
  }
}
