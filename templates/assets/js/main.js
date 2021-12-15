window.addEventListener("load", initslider);
var progress1 = document.getElementById("stp1");

function initslider(){
  var slider = document.getElementById("myRange");
  var output = document.getElementById("demo");
  
  var slider1 = document.getElementById("myRange1");
  var output1 = document.getElementById("demo1");
  
  var slider2 = document.getElementById("myRange2");
  var output2 = document.getElementById("demo2");

//  var slider2 = document.getElementById("myRange2");
//  var output2 = document.getElementById("demo2");

  var slider3 = document.getElementById("pred_dp");
  var output3 = document.getElementById("demo5");


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