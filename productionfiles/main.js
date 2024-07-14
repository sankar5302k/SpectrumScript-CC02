window.addEventListener('scroll', function() {
  const navbar = document.getElementById('navbar-example2');
  const cap1 = document.getElementById('cap1');

  if (window.scrollY > 70) {
      navbar.classList.add('scrolled');
      cap1.classList.add('scrolled');
  } else {
      navbar.classList.remove('scrolled');
      cap1.classList.remove('scrolled');

  }
});
let a = document.getElementsByClassName("prim1");
let b = document.getElementsByClassName("prim2");
for (let i = 0; i < a.length; i++) {
a[i].addEventListener('mouseover', () => {
for (let j = 0; j < b.length; j++) {
b[j].style.backgroundColor = "#cd61ffd7";
b[j].style.color = "white";

}

});
a[i].addEventListener('mouseout', () => {
for (let j = 0; j < b.length; j++) {
b[j].style.backgroundColor = "rgba(117, 70, 255, 0.87)";
b[j].style.color = "white";

}

});
}
for (let i = 0; i < b.length; i++) {
b[i].addEventListener('mouseover', () => {
for (let j = 0; j < b.length; j++) {
b[j].style.backgroundColor = "rgba(117, 70, 255, 0.87)";
b[j].style.color = "white";

}
});
b[i].addEventListener('mouseout', () => {
for (let j = 0; j < b.length; j++) {
b[j].style.backgroundColor = "rgba(117, 70, 255, 0.87)";
b[j].style.color = "white";

}
});
}
var i = 0;
var txt = 'Unveil the Hidden Hues with SpectrumScript !';
var speed = 50;
function typeWriter() {
if (i < txt.length) {
document.getElementById("demo").innerHTML += txt.charAt(i);
i++;
setTimeout(typeWriter, speed);
} else {
// Remove the cursor after the text is fully displayed
document.getElementById("demo").style.borderRight = 'none';
}
}

window.onload = typeWriter;
const myModal1 = document.getElementById('upload12')
const myInput1 = document.getElementById('upload1')

myModal2.addEventListener('shown.bs.modal', () => {
myInput2.focus()
})

document.getElementById('sub1').addEventListener('click', function(event) {
event.preventDefault(); 

const fileInput = document.getElementById('formFile');
if (fileInput.files.length > 0) {
window.location.href = './shadesout.html';

}
else{
alert("Enter the file")
}
});